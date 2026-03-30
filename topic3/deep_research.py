import asyncio
import uuid
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from openai import OpenAI, AsyncOpenAI
from tavily import TavilyClient, AsyncTavilyClient

class DeepResearchBot:
    def __init__(
        self,
        openai_client: AsyncOpenAI,
        tavily_client: AsyncTavilyClient,
        model: str = "Qwen/Qwen3-30B-A3B-Thinking-2507",
        max_queries: int = 5, # Max search queries in each iteration
        max_sources: int = 4, # Max number of sources fetched for each query
        max_iterations: int = 3, # Max number of search-analysis iterations
        search_depth: str = "advanced",
        verbose: bool = False
    ):
        """
        Initialize the Deep Research Bot.
        
        Args:
            openai_client: AsyncOpenAI client for LLM API calls
            tavily_client: TavilyClient for search operations
            model: Model name to use for reasoning (QwQ by default)
            max_sources: Maximum number of search results to retrieve per query
            search_depth: Tavily search depth ('basic' or 'advanced')
            verbose: Whether to print logs in real-time
        """
        self.openai_client = openai_client
        self.tavily_client = tavily_client
        self.model = model
        self.max_queries = max_queries
        self.max_sources = max_sources
        self.max_iterations = max_iterations
        self.search_depth = search_depth
        self.verbose = verbose
        
        # Store all research sessions: {(user_id, search_id): session_data}
        self.research_sessions = {}
    
    def _log(self, user_id: str, search_id: str, step: str, data: Any) -> None:
        """
        Log data to the research session and optionally print if verbose.
        
        Args:
            user_id: Unique ID of the user
            search_id: Unique ID of the search session
            step: Current step in the research process
            data: Data to log
        """
        session = self.research_sessions.get((user_id, search_id), {})
        if "logs" not in session:
            session["logs"] = []
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "data": data
        }
        
        session["logs"].append(log_entry)
        self.research_sessions[(user_id, search_id)] = session
        
        if self.verbose:
            print(f"[{step}] {datetime.now().isoformat()}")
            print(data)
            print("-" * 50)
    
    async def start_research(self, user_id: str, query: str) -> Dict[str, Any]:
        """
        Start a new research session with an initial query.
        
        Args:
            user_id: Unique ID of the user
            query: The initial research query
        
        Returns:
            Dict containing search_id and initial bot response
        """
        # Generate a unique search ID
        search_id = str(uuid.uuid4())
        
        # Initialize the research session
        self.research_sessions[(user_id, search_id)] = {
            "initial_query": query,
            "status": "started",
            "timestamp_start": datetime.now().isoformat(),
            "clarification_needed": True,
            "search_queries": [],
            "search_results": [],
            "analysis": [],
            "complete": False,
            "report": None,
            "logs": []
        }
        
        self._log(user_id, search_id, "start_research", {
            "query": query,
            "search_id": search_id
        })
        
        # Ask clarifying questions first
        clarification_response = await self._ask_clarifying_questions(user_id, search_id, query)
        
        return {
            "search_id": search_id,
            "response": clarification_response
        }
    
    async def _ask_clarifying_questions(self, user_id: str, search_id: str, query: str) -> str:
        """
        Have QwQ ask clarifying questions about the research query.
        
        Args:
            user_id: Unique ID of the user
            search_id: Unique ID of the search session
            query: The research query
        
        Returns:
            String containing the bot's clarifying questions
        """
        system_prompt = """You are a research assistant that helps users with in-depth research.
Your first task is to ask clarifying questions to better understand what the user wants to research.
Ask 1-3 specific questions that would help you understand the query better and perform a more targeted search.
Keep your questions concise and focused on key aspects like:
- Scope of research
- Specific areas of interest
- Time period relevance
- Required depth of information
- Any specific perspectives they want to explore"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"I need to research the following topic: {query}"}
        ]
        
        try:
            response = await self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7
            )
            clarifying_questions = response.choices[0].message.content
            
            self._log(user_id, search_id, "clarifying_questions", {
                "messages": messages,
                "response": clarifying_questions
            })
            
            return clarifying_questions
        except Exception as e:
            error_msg = f"Error generating clarifying questions: {str(e)}"
            self._log(user_id, search_id, "error", error_msg)
            return "I'd like to help with your research, but I need to ask a few questions first to better understand what you're looking for. Could you provide more details about your topic?"
    
    async def _process_user_response(self, user_id: str, search_id: str, response: str) -> Dict[str, Any]:
        """
        Process user's response to clarifying questions or additional information.
        
        Args:
            user_id: Unique ID of the user
            search_id: Unique ID of the search session
            response: User's response to the clarifying questions
        
        Returns:
            Dict containing the bot's response and status information
        """
        if (user_id, search_id) not in self.research_sessions:
            return {"error": "Research session not found", "search_id": search_id}
        
        session = self.research_sessions[(user_id, search_id)]
        
        # Add user response to logs
        self._log(user_id, search_id, "user_response", response)
        
        # If this is the first response, it's answering clarifying questions
        if session.get("clarification_needed", True):
            session["clarification_needed"] = False
            session["user_clarification"] = response
        else:
            # This is additional information after initial clarification
            session["follow_up_responses"] = session.get("follow_up_responses", []) + [response]
            # Reset completion status to force new iterations
            session["complete"] = False
        
        # Save updated session
        self.research_sessions[(user_id, search_id)] = session
        
        # Whether this is a follow-up depends on whether we just processed clarification
        is_follow_up = not session.get("clarification_needed", False)
        
        # Run the iterative search process
        return await self._run_iterative_search(user_id, search_id, is_follow_up)
    
    async def _formulate_search_queries(self, user_id: str, search_id: str, follow_up: bool = False) -> List[str]:
        """
        Have QwQ formulate search queries based on the research topic and clarifications.
        
        Args:
            user_id: Unique ID of the user
            search_id: Unique ID of the search session
            follow_up: Whether this is a follow-up query formulation
        
        Returns:
            List of search queries
        """
        session = self.research_sessions[(user_id, search_id)]
        initial_query = session["initial_query"]
        
        if follow_up:
            # If this is a follow-up, we consider previous search results and analyses
            system_prompt = f"""You are a research assistant helping with in-depth research.
    
    Based on the initial query, user clarifications, and prior search results, generate up to {self.max_queries} NEW and highly specific search queries 
    that will help gather additional information to complete the research.
    
    Your response must follow this exact format:
    <search_queries>
    1. [First search query]
    2. [Second search query]
    3. [Third search query]
    4. [Fourth search query]
    </search_queries>
    
    The queries should:
    - Be specific and focused on filling information gaps identified in the analysis
    - Not overlap with previous queries
    - Be phrased as search engine queries (not questions)
    - Be concise but detailed enough to find relevant information"""
    
            previous_queries = "\n".join([f"- {q}" for q in session.get("search_queries", [])])
            previous_analysis = session.get("analysis", [])
            last_analysis = previous_analysis[-1] if previous_analysis else "No previous analysis"
            
            user_content = f"""Initial query: {initial_query}
    User clarification: {session.get('user_clarification', 'None provided')}
    Follow-up responses: {session.get('follow_up_responses', ['None provided'])}
    
    Previous search queries:
    {previous_queries}
    
    Last analysis of results:
    {last_analysis}
    
    Based on the above, formulate new search queries to find missing information."""
        else:
            # Initial query formulation based on the original query and clarifications
            system_prompt = f"""You are a research assistant helping with in-depth research.
    
    Based on the initial query and user clarifications, generate up to {self.max_queries} effective search queries that will gather comprehensive information on the topic.
    
    Your response must follow this exact format:
    <search_queries>
    1. [First search query]
    2. [Second search query]
    3. [Third search query]
    4. [Fourth search query]
    5. [Fifth search query]
    </search_queries>
    
    The queries should:
    - Be specific and focused on different aspects of the research question
    - Be phrased as search engine queries (not questions)
    - Be concise but detailed enough to find relevant information"""
    
            user_content = f"""Initial query: {initial_query}
    User clarification: {session.get('user_clarification', 'None provided')}
    
    Based on the above, formulate effective search queries to gather comprehensive information."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        try:
            response = await self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7
            )
            
            # Extract queries from the XML-like tags
            query_text = response.choices[0].message.content
            
            # Extract content between <search_queries> tags
            import re
            queries_match = re.search(r'<search_queries>(.*?)</search_queries>', query_text, re.DOTALL)
            
            if queries_match:
                queries_content = queries_match.group(1).strip()
                # Extract numbered queries
                number_pattern = r'\d+\.\s*(.*?)(?=\n\d+\.|\n*$)'
                matches = re.findall(number_pattern, queries_content, re.DOTALL)
                
                if matches:
                    queries = [q.strip() for q in matches if q.strip()]
                else:
                    # Fallback to line-by-line parsing if numbered pattern fails
                    lines = queries_content.split('\n')
                    queries = [line.strip() for line in lines if line.strip() and not line.strip().isdigit()]
            else:
                # If tags not found, try to extract numbered list directly
                number_pattern = r'\d+\.\s*(.*?)(?=\n\d+\.|\n*$)'
                matches = re.findall(number_pattern, query_text, re.DOTALL)
                
                if matches:
                    queries = [q.strip() for q in matches if q.strip()]
                else:
                    # Last resort: split by newlines and clean up
                    query_lines = query_text.split('\n')
                    queries = []
                    for line in query_lines:
                        line = line.strip()
                        if line and not line.startswith('<') and not line.endswith('>'):
                            # Remove any numbering at the beginning
                            clean_line = re.sub(r'^\d+\.\s*', '', line)
                            if clean_line:
                                queries.append(clean_line)
            
            # Keep only unique queries, limit to 5 max
            unique_queries = []
            for q in queries:
                if q not in unique_queries and q not in session.get("search_queries", []):
                    unique_queries.append(q)
                    if len(unique_queries) >= 5:
                        break
            
            # Update session with new queries
            session["search_queries"] = session.get("search_queries", []) + unique_queries
            self.research_sessions[(user_id, search_id)] = session
            
            self._log(user_id, search_id, "formulated_queries", {
                "messages": messages,
                "response": query_text,
                "parsed_queries": unique_queries
            })
            
            return unique_queries
        except Exception as e:
            error_msg = f"Error formulating search queries: {str(e)}"
            self._log(user_id, search_id, "error", error_msg)
            return ["Error generating queries"]

    async def _run_iterative_search(self, user_id: str, search_id: str, is_follow_up: bool) -> Dict[str, Any]:
        """
        Run the iterative search process until completion or max iterations.
        
        Args:
            user_id: Unique ID of the user
            search_id: Unique ID of the search session
            is_follow_up: Whether this is a follow-up to initial clarification
            
        Returns:
            Dict containing the bot's response and status information
        """
        iteration_count = 0
        is_complete = False
        
        # Run iterations until research is complete or self.max_iterations is reached
        while not is_complete and iteration_count < self.max_iterations:
            self._log(user_id, search_id, "iteration_start", {
                "iteration_number": iteration_count + 1,
                "is_follow_up": is_follow_up
            })
            
            # Run a single search iteration
            is_complete = await self._run_search_iteration(user_id, search_id, is_follow_up)
            
            # Increment counter
            iteration_count += 1
            
            # Get updated session status
            session = self.research_sessions[(user_id, search_id)]
            is_complete = session.get("complete", False)
        
        # Generate appropriate response based on completion status
        session = self.research_sessions[(user_id, search_id)]
        
        if is_complete:
            report = await self._generate_report(user_id, search_id)
            return {
                "status": "complete",
                "response": f"I've completed my research after {iteration_count} search iterations and prepared a report for you:",
                "report": report,
                "search_id": search_id,
                "iterations": iteration_count
            }
        else:
            # Research is not complete, but we've reached max iterations
            # Generate a partial report with what we have
            report = await self._generate_report(user_id, search_id, is_partial=True)
            return {
                "status": "partial_complete",
                "response": f"I've conducted {iteration_count} search iterations but couldn't find all the information. Here's what I was able to discover:",
                "report": report,
                "search_id": search_id,
                "iterations": iteration_count
            }

    async def _run_search_iteration(self, user_id: str, search_id: str, follow_up: bool = False) -> bool:
        """
        Run a single iteration of the search process.
        
        Args:
            user_id: Unique ID of the user
            search_id: Unique ID of the search session
            follow_up: Whether this is a follow-up iteration
            
        Returns:
            Boolean indicating whether the research is complete
        """
        session = self.research_sessions[(user_id, search_id)]
        
        try:
            # Step 1: Formulate search queries
            search_queries = await self._formulate_search_queries(user_id, search_id, follow_up=follow_up)
            
            if not search_queries or search_queries == ["Error generating queries"]:
                self._log(user_id, search_id, "iteration_error", "Failed to generate search queries")
                return False
            
            # Step 2: Execute searches
            search_results = await self._perform_searches(user_id, search_id, search_queries)
            
            if not search_results:
                self._log(user_id, search_id, "iteration_error", "Failed to get search results")
                return False
            
            # Step 3: Analyze search results
            analysis, is_complete = await self._analyze_search_results(user_id, search_id)
            
            # Update session with completion status
            session["complete"] = is_complete
            self.research_sessions[(user_id, search_id)] = session
            
            return is_complete
        
        except Exception as e:
            self._log(user_id, search_id, "iteration_error", str(e))
            return False
    
    async def _perform_searches(self, user_id: str, search_id: str, queries: List[str]) -> List[Dict]:
        """
        Perform Tavily searches for each query.
        
        Args:
            user_id: Unique ID of the user
            search_id: Unique ID of the search session
            queries: List of search queries
        
        Returns:
            List of search results
        """
        all_results = []
        
        for query in queries:
            try:
                results = await self.tavily_client.search(
                    query=query,
                    search_depth=self.search_depth,
                    max_results=self.max_sources
                )
                
                # Add the query to the results for reference
                results["query"] = query
                
                all_results.append(results)
                
                self._log(user_id, search_id, "search_results", {
                    "query": query,
                    "results": results
                })
                
                # Introduce a small delay to avoid rate limiting
                await asyncio.sleep(0.5)
            except Exception as e:
                error_msg = f"Error performing search for query '{query}': {str(e)}"
                self._log(user_id, search_id, "error", error_msg)
        
        # Update session with new search results
        session = self.research_sessions[(user_id, search_id)]
        session["search_results"] = session.get("search_results", []) + all_results
        self.research_sessions[(user_id, search_id)] = session
        
        return all_results
    
    async def _analyze_search_results(self, user_id: str, search_id: str) -> Tuple[str, bool]:
        """
        Have QwQ analyze search results and determine if more searches are needed.
        
        Args:
            user_id: Unique ID of the user
            search_id: Unique ID of the search session
        
        Returns:
            Tuple of (analysis_text, is_complete)
        """
        session = self.research_sessions[(user_id, search_id)]
        initial_query = session["initial_query"]
        search_results = session.get("search_results", [])
        previous_analyses = session.get("analysis", [])
        
        # Get the most recent analysis if available
        previous_analysis = previous_analyses[-1] if previous_analyses else ""
        
        # Format search results for QwQ
        formatted_results = []
        
        for result_set in search_results:
            query = result_set.get("query", "Unknown query")
            formatted_results.append(f"## Query: {query}")
            
            for i, result in enumerate(result_set.get("results", [])):
                content = result.get("content", "").strip()
                url = result.get("url", "No URL")
                
                formatted_results.append(f"### Result {i+1}")
                formatted_results.append(f"Source: {url}")
                formatted_results.append(f"Content: {content}")
                formatted_results.append("")
        
        all_results_text = "\n".join(formatted_results)
        
        system_prompt = """You are a research assistant analyzing search results to determine if they sufficiently answer a research query.
    
    Analyze the search results provided to determine:
    1. What valuable information has been found
    2. What important information is still missing
    3. Whether the research can be considered complete
    
    Your analysis should include:
    - A summary of key findings from the search results
    - Identification of any contradictions or gaps in the information
    - A clear indication whether more research is needed
    
    End your analysis with EITHER:
    <complete> - if you believe the research query can be sufficiently answered given the search results available so far
    <incomplete> - if more information is needed (and specify what information)"""
    
        # Add context about previous analysis if available
        previous_analysis_context = ""
        if previous_analysis:
            previous_analysis_context = f"""
    Previous Analysis:
    {previous_analysis}
    
    Your task is to analyze the newly found information in conjunction with the previous analysis.
    Focus on whether the new search results fill the gaps identified in the previous analysis.
    Avoid repeating observations already made in the previous analysis unless you have new insights.
    """
    
        user_content = f"""Initial research query: {initial_query}
    User clarification: {session.get('user_clarification', 'None provided')}
    {previous_analysis_context}
    Search Results:
    {all_results_text}
    
    Based on these results, analyze whether we have sufficient information to answer the research query."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        try:
            response = await self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=16384
            )
            
            analysis = response.choices[0].message.content
            
            # Determine if the research is complete
            is_complete = "<complete>" in analysis.lower()
            
            self._log(user_id, search_id, "analysis", {
                "analysis": analysis,
                "is_complete": is_complete,
                "had_previous_analysis": bool(previous_analysis)
            })
            
            # Update session with analysis
            session["analysis"] = session.get("analysis", []) + [analysis]
            self.research_sessions[(user_id, search_id)] = session
            
            return analysis, is_complete
        except Exception as e:
            error_msg = f"Error analyzing search results: {str(e)}"
            self._log(user_id, search_id, "error", error_msg)
            return "I'm having trouble analyzing the search results. Let's try a different approach.", False
    
    async def _generate_report(self, user_id: str, search_id: str, is_partial: bool = False) -> str:
        """
        Generate a final markdown report based on all collected information.
        
        Args:
            user_id: Unique ID of the user
            search_id: Unique ID of the search session
            is_partial: Whether this is a partial report due to incomplete research
            
        Returns:
            String containing the final report in markdown format
        """
        session = self.research_sessions[(user_id, search_id)]
        initial_query = session["initial_query"]
        search_results = session.get("search_results", [])
        
        # Format search results for QwQ
        formatted_results = []
        
        for result_set in search_results:
            query = result_set.get("query", "Unknown query")
            formatted_results.append(f"## Query: {query}")
            
            for i, result in enumerate(result_set.get("results", [])):
                content = result.get("content", "").strip()
                url = result.get("url", "No URL")
                
                formatted_results.append(f"### Result {i+1}")
                formatted_results.append(f"Source: {url}")
                formatted_results.append(f"Content: {content}")
                formatted_results.append("")
        
        all_results_text = "\n".join(formatted_results)
        
        system_prompt = """You are a research assistant creating a comprehensive research report based on collected information.
    
    Create a well-structured, detailed markdown report that thoroughly addresses the research query.
    
    Your report should include:
    1. Executive Summary
    2. Key Findings (organized by themes or sub-topics)
    3. Analysis and Discussion
    4. Conclusion
    5. References (properly formatted and cited)
    
    Guidelines:
    - Format the report in clean, proper markdown
    - Ensure that your report answers user's question
    - Include headings, subheadings, bullet points, and other formatting for readability
    - Synthesize information from multiple sources rather than quoting directly
    - Remain objective and balanced in your analysis
    - Cite sources appropriately throughout the report using footnotes or endnotes
    - Ensure the report is comprehensive but concise"""
    
        # If this is a partial report, modify the prompt
        if is_partial:
            system_prompt += """
    
    IMPORTANT: This is a partial report based on incomplete research. Please:
    - Clearly indicate in the executive summary that this is a partial report
    - Note any significant information gaps in a dedicated section
    - Suggest what additional research might be needed
    - Focus on presenting the available information accurately rather than making unsupported claims"""
    
        user_content = f"""Research Query: {initial_query}
    User clarification: {session.get('user_clarification', 'None provided')}
    
    Search Results:
    {all_results_text}
    
    Based on this information, create a {'comprehensive' if not is_partial else 'partial'} research report in markdown format."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        try:
            response = await self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=32768  # Ensure we have enough tokens for a detailed report
            )
            
            report = response.choices[0].message.content
            
            self._log(user_id, search_id, "report", {
                "report": report,
                "is_partial": is_partial
            })
            
            # Update session with the report
            session["report"] = report
            session["timestamp_complete"] = datetime.now().isoformat()
            session["status"] = "completed" if not is_partial else "partially_completed"
            self.research_sessions[(user_id, search_id)] = session
            
            return report
        except Exception as e:
            error_msg = f"Error generating report: {str(e)}"
            self._log(user_id, search_id, "error", error_msg)
            return "I apologize, but I encountered an error while generating your report. Please try again later."
    
    def get_session_data(self, user_id: str, search_id: str) -> Dict[str, Any]:
        """
        Retrieve the complete session data for a specific research session.
        
        Args:
            user_id: Unique ID of the user
            search_id: Unique ID of the search session
        
        Returns:
            Dict containing all session data
        """
        if (user_id, search_id) not in self.research_sessions:
            return {"error": "Research session not found"}
        
        return self.research_sessions[(user_id, search_id)]

    async def chat(self, user_id: str, message: str) -> Dict[str, Any]:
        """
        Main entry point for user interaction with the research bot.
        This function routes the user's message to the appropriate handler based on context.
        
        Args:
            user_id: Unique ID of the user
            message: User's message
            
        Returns:
            Dict containing the bot's response and other relevant information
        """
        # Check if this is a new message or continuation of existing research
        active_sessions = [session_id for (uid, session_id), session in self.research_sessions.items() 
                          if uid == user_id and session.get("status") not in ["completed", "abandoned"]]
        
        # If no active sessions or message seems like a new query, start new research
        if not active_sessions:
            response = await self.start_research(user_id, message)
            return response
        
        # Get the most recent active session
        search_id = active_sessions[-1]  # Use the most recent session
        session = self.research_sessions.get((user_id, search_id), {})
        
        # Process the message as a response to the current research session
        return await self._process_user_response(user_id, search_id, message)
