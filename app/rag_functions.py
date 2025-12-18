"""
RAG Pipeline Functions
Contains all the core RAG functionality extracted from the notebook
"""

import pandas as pd
import os
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


# Configuration
K_DOCUMENTS = 5
CHROMA_DIR = "./chroma_db"


# Pydantic Models
class CompanyExtraction(BaseModel):
    company_names: List[str] = Field(
        default_factory=list,
        description="Company names from the valid list. Empty if none found."
    )


class ExpandedQueries(BaseModel):
    queries: List[str] = Field(
        ...,
        description="1-3 expanded queries rephrasing the original from different perspectives"
    )


class RAGPipeline:
    def __init__(self, api_key: str, csv_path: str = "Data Manipulation/companies_enriched_deepseek.csv", chroma_dir: str = CHROMA_DIR):
        """Initialize the RAG pipeline with API key and data"""
        os.environ["OPENAI_API_KEY"] = api_key
        
        # Load data
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at: {csv_path}")
        
        self.df = pd.read_csv(csv_path)
        self.VALID_COMPANIES = self.df['companyName'].unique().tolist()
        
        # Initialize LLM
        self.llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
        
        # Load existing vector store
        self.vectorstore = None
        self._load_vector_store(chroma_dir)
        
        # Setup chains
        self._setup_chains()
    
    def _load_vector_store(self, chroma_dir: str):
        """Load existing vector database"""
        if not os.path.exists(chroma_dir):
            raise FileNotFoundError(f"Vector database not found at: {chroma_dir}")
        
        embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma(
            persist_directory=chroma_dir,
            embedding_function=embeddings
        )
    
    def _setup_chains(self):
        """Setup all LLM chains"""
        # Company extraction chain
        company_parser = JsonOutputParser(pydantic_object=CompanyExtraction)
        company_extraction_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Your task is to extract company names explicitly mentioned in the user query.\n\n"
                "Rules:\n"
                "- Do NOT infer, guess, or predict company names.\n"
                "- Only extract a company if it is explicitly mentioned in the query.\n"
                "- Only return companies that exist in the following list: {company_list}.\n"
                "- If the query mentions a partial, lowercase, or variant form of a company name "
                "(e.g., 'traton'), map it to the exact company name as written in the company list "
                "(e.g., 'Traton SE').\n"
                "- If no company from the list is explicitly mentioned, return an empty list.\n\n"
                "{format_instructions}"
            ),
            ("human", "{query}")
        ]).partial(
            format_instructions=company_parser.get_format_instructions(),
            company_list=str(self.VALID_COMPANIES)
        )
        self.company_extraction_chain = company_extraction_prompt | self.llm | company_parser
        
        # Query expansion chain
        query_expansion_parser = JsonOutputParser(pydantic_object=ExpandedQueries)
        query_expansion_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Your task is to expand the original user query into **1-3 distinct**, non-overlapping "
                "sub-queries by decomposing it into smaller, well-defined components.\n\n"
                "IMPORTANT: Generate as FEW queries (minimum 1, maximum 3). Only create "
                "multiple queries if the original question has multiple distinct aspects.\n\n"
                "Each generated sub-query must focus on a DIFFERENT aspect, condition, or constraint "
                "explicitly implied by the original query.\n\n"
                "Do NOT generate multiple sub-queries with the same meaning.\n\n"
                "The generated sub-queries will be used for semantic search. Therefore:\n"
                "- Make each sub-query long, detailed, and information-dense.\n"
                "- Preserve the original intent exactly.\n"
                "- Include all relevant context from the original query in each sub-query where applicable.\n\n"
                "RULES:\n"
                "1. Generate between 1 and 3 sub-queries (MAXIMUM of 3, prefer fewer).\n"
                "2. Do NOT introduce new facts, entities, industries, or constraints that are not implied "
                "by the original query. For instance don't try to guess the company industry or ptoducts by its name.\n"
                "3. Each sub-query must represent a unique aspect or constraint of the original query.\n"
                "4. If the original query is simple and focused, generate only 1 query.\n"
                " Never ever guess industry and products by the name of company! If the user didn't mention the industry"
                "and product, never add them in the queries.\n"
                "5. If multiple conditions are implied (e.g., sector A AND sector B), separate them into "
                "individual sub-queries.\n\n"
                "{format_instructions}"
            ),
            ("human", "Original query: {query}")
        ]).partial(
            format_instructions=query_expansion_parser.get_format_instructions()
        )
        self.query_expansion_chain = query_expansion_prompt | self.llm | query_expansion_parser
        
        # Answer generation chain
        answer_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a precise QA assistant. CRITICAL:\n"
                "1. Answer ONLY from the provided context\n"
                "2. No external knowledge\n"
                "3. Say 'Cannot answer' if context insufficient\n"
                "4. Be specific and cite details\n\n"
                "Context:\n{context}"
            ),
            ("human", "{question}")
        ])
        self.answer_chain = answer_prompt | self.llm | StrOutputParser()
    
    def contextual_retriever(self, queries: List[str], k: int = K_DOCUMENTS) -> List[Document]:
        """Retrieve documents based on queries with company filtering"""
        retrieved_docs = []
        seen_contents = set()
        
        for query in queries:
            extraction = self.company_extraction_chain.invoke({"query": query})
            companies = extraction.get('company_names', [])
            
            if companies:
                for company in companies:
                    retriever = self.vectorstore.as_retriever(
                        search_kwargs={"filter": {"company_name": company}, "k": k}
                    )
                    docs = retriever.invoke(query)
                    for doc in docs:
                        if doc.page_content not in seen_contents:
                            seen_contents.add(doc.page_content)
                            retrieved_docs.append(doc)
            else:
                retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
                docs = retriever.invoke(query)
                for doc in docs:
                    if doc.page_content not in seen_contents:
                        seen_contents.add(doc.page_content)
                        retrieved_docs.append(doc)
        
        return retrieved_docs
    
    def query(self, user_query: str) -> Dict[str, Any]:
        """Main RAG pipeline function"""
        # Expand query
        expansion = self.query_expansion_chain.invoke({"query": user_query})
        expanded_queries = expansion['queries']
        
        # Retrieve documents
        all_queries = [user_query] + expanded_queries
        docs = self.contextual_retriever(all_queries)
        
        # Format context
        context = "\n\n---\n\n".join(d.page_content for d in docs)
        
        # Generate answer
        answer = self.answer_chain.invoke({"context": context, "question": user_query})
        
        # Get company names
        companies = list(set(d.metadata['company_name'] for d in docs))
        
        return {
            "answer": answer,
            "expanded_queries": expanded_queries,
            "num_docs": len(docs),
            "companies": companies,
            "context_length": len(context)
        }


class RAGEvaluator:
    def __init__(self, rag_pipeline: RAGPipeline):
        """Initialize evaluator with RAG pipeline"""
        self.rag_pipeline = rag_pipeline
        self.judge_llm = rag_pipeline.llm
        
        # Setup judge chain
        judge_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an evaluation judge. Compare the expected answer with the predicted answer.\n"
                "Determine if the predicted answer is correct (captures the same meaning/information).\n\n"
                "Format your response as:\n"
                "Decision: True/False\n"
                "Reason: [One short sentence explaining why]"
            ),
            (
                "human",
                "Question: {question}\n\n"
                "Expected Answer: {expected}\n\n"
                "Predicted Answer: {predicted}\n\n"
                "Evaluate if the predicted answer is correct:"
            )
        ])
        self.judge_chain = judge_prompt | self.judge_llm | StrOutputParser()
    
    def evaluate(self, test_df: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate RAG system using test dataframe"""
        results = []
        correct_count = 0
        total_count = len(test_df)
        
        for idx, row in test_df.iterrows():
            question = row['question']
            expected = row['expected_answer']
            
            # Get RAG prediction
            rag_result = self.rag_pipeline.query(question)
            predicted = rag_result['answer']
            
            # Get LLM judge verdict
            verdict = self.judge_chain.invoke({
                "question": question,
                "expected": expected,
                "predicted": predicted
            })
            
            # Parse verdict and reason
            lines = verdict.strip().split('\n')
            decision_line = lines[0] if len(lines) > 0 else ""
            reason_line = lines[1] if len(lines) > 1 else ""
            
            is_correct = "true" in decision_line.lower()
            reason = reason_line.replace("Reason:", "").strip()
            
            results.append({
                'question': question,
                'expected_answer': expected,
                'rag_answer': predicted,
                'decision': is_correct,
                'reason': reason
            })
            
            if is_correct:
                correct_count += 1
        
        accuracy = correct_count / total_count
        results_df = pd.DataFrame(results)
        
        return {
            "accuracy": accuracy,
            "correct": correct_count,
            "total": total_count,
            "results_df": results_df
        }