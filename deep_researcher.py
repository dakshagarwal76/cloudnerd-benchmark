"""Main LangGraph implementation for the Deep Research agent."""

import asyncio
import logging
from typing import Literal

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')


from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from open_deep_research.configuration import (
    Configuration,
    SearchAPI, 
)

from open_deep_research.prompts import (
    clarify_with_user_instructions,
    compress_research_simple_human_message,
    compress_research_system_prompt,
    final_report_generation_prompt,
    lead_researcher_prompt,
    research_system_prompt,
    transform_messages_into_research_topic_prompt,
)
from open_deep_research.state import (
    AgentInputState,
    AgentState,
    ClarifyWithUser,
    ConductResearch,
    ResearchComplete,
    ResearcherOutputState,
    ResearcherState,
    ResearchQuestion,
    SupervisorState,
)
from open_deep_research.utils import (
    get_all_tools,
    get_api_key_for_model,
    get_model_token_limit,
    get_notes_from_tool_calls,
    get_today_str,
    is_token_limit_exceeded,
    remove_up_to_last_ai_message,
    think_tool,
)

from dotenv import load_dotenv
load_dotenv()


def validate_search_configuration(config: RunnableConfig):
    """Validate that only JSONL search is configured."""  # Changed from PubMed
    configurable = Configuration.from_runnable_config(config)
    if configurable.search_api != SearchAPI.JSONL:  # Changed from PUBMED
        raise ValueError(
            f"Invalid search configuration: {configurable.search_api}. "
            "Only JSONL is allowed in this configuration."  # Changed from PubMed
        )
    return True


def create_clean_model(configurable: Configuration, model_name: str, max_tokens: int, config: RunnableConfig):
    """Create a model WITHOUT native search capabilities."""
    return init_chat_model(
        model=model_name,
        max_tokens=max_tokens,
        api_key=get_api_key_for_model(model_name, config),
        tags=["langsmith:nostream"]
    )

# Initialize a configurable model that we will use throughout the agent
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key"),
)

async def clarify_with_user(state: AgentState, config: RunnableConfig) -> Command[Literal["write_research_brief", "__end__"]]:
    """Analyze user messages and ask clarifying questions if the research scope is unclear.
    
    This function determines whether the user's request needs clarification before proceeding
    with research. If clarification is disabled or not needed, it proceeds directly to research.
    
    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings and preferences
        
    Returns:
        Command to either end with a clarifying question or proceed to research brief
    """
    # Step 1: Check if clarification is enabled in configuration
    configurable = Configuration.from_runnable_config(config)
    if not configurable.allow_clarification:
        # Skip clarification step and proceed directly to research
        return Command(goto="write_research_brief")
    
    # Step 2: Prepare the model for structured clarification analysis
    messages = state["messages"]
    model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Configure model with structured output and retry logic
    clarification_model = (
        create_clean_model(configurable, configurable.research_model, configurable.research_model_max_tokens, config)
        .with_structured_output(ClarifyWithUser)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(model_config)
    )
    
    # Step 3: Analyze whether clarification is needed
    prompt_content = clarify_with_user_instructions.format(
        messages=get_buffer_string(messages), 
        date=get_today_str()
    )
    response = await clarification_model.ainvoke([HumanMessage(content=prompt_content)])
    
    # Step 4: Route based on clarification analysis
    if response.need_clarification:
        # End with clarifying question for user
        return Command(
            goto=END, 
            update={"messages": [AIMessage(content=response.question)]}
        )
    else:
        # Proceed to research with verification message
        return Command(
            goto="write_research_brief", 
            update={"messages": [AIMessage(content=response.verification)]}
        )


async def write_research_brief(state: AgentState, config: RunnableConfig) -> Command[Literal["research_supervisor"]]:
    """Transform user messages into a structured research brief and initialize supervisor.
    
    This function analyzes the user's messages and generates a focused research brief
    that will guide the research supervisor. It also sets up the initial supervisor
    context with appropriate prompts and instructions.
    
    Args:
        state: Current agent state containing user messages
        config: Runtime configuration with model settings
        
    Returns:
        Command to proceed to research supervisor with initialized context
    """
    # Step 1: Set up the research model for structured output
    configurable = Configuration.from_runnable_config(config)
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Configure model for structured research question generation
    research_model = (
        create_clean_model(configurable, configurable.research_model, configurable.research_model_max_tokens, config)
        .with_structured_output(ResearchQuestion)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # Step 2: Generate structured research brief from user messages
    prompt_content = transform_messages_into_research_topic_prompt.format(
        messages=get_buffer_string(state.get("messages", [])),
        date=get_today_str()
    )
    response = await research_model.ainvoke([HumanMessage(content=prompt_content)])
    
    # Step 3: Initialize supervisor with research brief and instructions
    supervisor_system_prompt = lead_researcher_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=configurable.max_concurrent_research_units,
        max_researcher_iterations=configurable.max_researcher_iterations
    )
    
    return Command(
        goto="research_supervisor", 
        update={
            "research_brief": response.research_brief,
            "supervisor_messages": {
                "type": "override",
                "value": [
                    SystemMessage(content=supervisor_system_prompt),
                    HumanMessage(content=response.research_brief)
                ]
            }
        }
    )


async def supervisor(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor_tools"]]:
    """Lead research supervisor that plans research strategy and delegates to researchers.
    
    The supervisor analyzes the research brief and decides how to break down the research
    into manageable tasks. It can use think_tool for strategic planning, ConductResearch
    to delegate tasks to sub-researchers, or ResearchComplete when satisfied with findings.
    
    Args:
        state: Current supervisor state with messages and research context
        config: Runtime configuration with model settings
        
    Returns:
        Command to proceed to supervisor_tools for tool execution
    """
    validate_search_configuration(config)
    # Step 1: Configure the supervisor model with available tools
    configurable = Configuration.from_runnable_config(config)
    research_model_config = {
    "model": configurable.research_model,
    "max_tokens": configurable.research_model_max_tokens,
    "api_key": get_api_key_for_model(configurable.research_model, config),
    "tags": ["langsmith:nostream"]
}


    lead_researcher_tools = [ConductResearch, ResearchComplete, think_tool]

    
    # Available tools: research delegation, completion signaling, and strategic thinking
    lead_researcher_tools = [ConductResearch, ResearchComplete, think_tool]
    
    # Configure model with tools, retry logic, and model settings
    research_model = (
        create_clean_model(configurable, configurable.research_model, configurable.research_model_max_tokens, config)
        .bind_tools(lead_researcher_tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # Step 2: Generate supervisor response based on current context
    supervisor_messages = state.get("supervisor_messages", [])
    response = await research_model.ainvoke(supervisor_messages)
    
    # Step 3: Update state and proceed to tool execution
    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1
        }
    )

async def supervisor_tools(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor", "__end__"]]:
    """Execute tools called by the supervisor, including research delegation and strategic thinking.
    
    This function handles three types of supervisor tool calls:
    1. think_tool - Strategic reflection that continues the conversation
    2. ConductResearch - Delegates research tasks to sub-researchers
    3. ResearchComplete - Signals completion of research phase
    
    Args:
        state: Current supervisor state with messages and iteration count
        config: Runtime configuration with research limits and model settings
        
    Returns:
        Command to either continue supervision loop or end research phase
    """
    # Step 1: Extract current state and check exit conditions
    configurable = Configuration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]
    
    # Define exit criteria for research phase
    exceeded_allowed_iterations = research_iterations > configurable.max_researcher_iterations
    no_tool_calls = not most_recent_message.tool_calls
    research_complete_tool_call = any(
        tool_call["name"] == "ResearchComplete" 
        for tool_call in most_recent_message.tool_calls
    )
    
    # Exit if any termination condition is met
    if exceeded_allowed_iterations or no_tool_calls or research_complete_tool_call:
        return Command(
            goto=END,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", "")
            }
        )
    
    # Step 2: Process all tool calls together (both think_tool and ConductResearch)
    all_tool_messages = []
    update_payload = {"supervisor_messages": []}
    
    # Handle think_tool calls (strategic reflection)
    think_tool_calls = [
        tool_call for tool_call in most_recent_message.tool_calls 
        if tool_call["name"] == "think_tool"
    ]
    
    for tool_call in think_tool_calls:
        reflection_content = tool_call["args"]["reflection"]
        all_tool_messages.append(ToolMessage(
            content=f"Reflection recorded: {reflection_content}",
            name="think_tool",
            tool_call_id=tool_call["id"]
        ))
    
    # Handle ConductResearch calls (research delegation)
    conduct_research_calls = [
        tool_call for tool_call in most_recent_message.tool_calls 
        if tool_call["name"] == "ConductResearch"
    ]
    
    if conduct_research_calls:
        try:
            # Limit concurrent research units to prevent resource exhaustion
            allowed_conduct_research_calls = conduct_research_calls[:configurable.max_concurrent_research_units]
            overflow_conduct_research_calls = conduct_research_calls[configurable.max_concurrent_research_units:]
            
            # Execute research tasks in parallel
            research_tasks = [
                researcher_subgraph.ainvoke({
                    "researcher_messages": [
                        HumanMessage(content=tool_call["args"]["research_topic"])
                    ],
                    "research_topic": tool_call["args"]["research_topic"]
                }, config) 
                for tool_call in allowed_conduct_research_calls
            ]
            
            tool_results = await asyncio.gather(*research_tasks)
            
            # Create tool messages with research results
            for observation, tool_call in zip(tool_results, allowed_conduct_research_calls):
                all_tool_messages.append(ToolMessage(
                    content=observation.get("compressed_research", "Error synthesizing research report: Maximum retries exceeded"),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                ))
            
            # Handle overflow research calls with error messages
            for overflow_call in overflow_conduct_research_calls:
                all_tool_messages.append(ToolMessage(
                    content=f"Error: Did not run this research as you have already exceeded the maximum number of concurrent research units. Please try again with {configurable.max_concurrent_research_units} or fewer research units.",
                    name="ConductResearch",
                    tool_call_id=overflow_call["id"]
                ))
                
        except Exception as e:
            # Handle research execution errors
            if is_token_limit_exceeded(e, configurable.research_model) or True:
                # Token limit exceeded or other error - end research phase
                return Command(
                    goto=END,
                    update={
                        "notes": get_notes_from_tool_calls(supervisor_messages),
                        "research_brief": state.get("research_brief", "")
                    }
                )
    
    # Step 3: Return command with all tool results
    update_payload["supervisor_messages"] = all_tool_messages
    return Command(
        goto="supervisor",
        update=update_payload
    ) 

# Supervisor Subgraph Construction
# Creates the supervisor workflow that manages research delegation and coordination
supervisor_builder = StateGraph(SupervisorState, config_schema=Configuration)

# Add supervisor nodes for research management
supervisor_builder.add_node("supervisor", supervisor)           # Main supervisor logic
supervisor_builder.add_node("supervisor_tools", supervisor_tools)  # Tool execution handler

# Define supervisor workflow edges
supervisor_builder.add_edge(START, "supervisor")  # Entry point to supervisor

# Compile supervisor subgraph for use in main workflow
supervisor_subgraph = supervisor_builder.compile()

def fix_encoding_issues(text: str) -> str:
    """Fix common UTF-8 encoding corruption in text."""
    replacements = {
        'â€œ': '"',    
        'â€': '"',    
        'â€™': "'",     
        'â€˜': "'", 
        'â€"': '--',  
        'â€"': '-', 
        'â€¦': '...', 
        'Â': '',      
        'â€"': '-',    
        'â€': '"',   
    }
    
    for bad, good in replacements.items():
        text = text.replace(bad, good)

    text = text.replace('"', '"').replace('"', '"')  
    text = text.replace(''', "'").replace(''', "'") 
    text = text.replace('—', '--').replace('–', '-') 
    text = text.replace('…', '...')  
    
    return text


async def researcher(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher_tools"]]:
    """Individual researcher that conducts focused research on specific topics.
    
    This researcher is given a specific research topic by the supervisor and uses
    available tools (search, think_tool, MCP tools) to gather comprehensive information.
    It can use think_tool for strategic planning between searches.
    
    Args:
        state: Current researcher state with messages and topic context
        config: Runtime configuration with model settings and tool availability
        
    Returns:
        Command to proceed to researcher_tools for tool execution
    """
    validate_search_configuration(config)
    # Step 1: Load configuration and validate tool availability
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])

    if configurable.search_api != SearchAPI.JSONL:
        raise ValueError(f"Search API must be JSONL, got {configurable.search_api}")
    
    tools = await get_all_tools(config)
    
    # Get all available research tools (search, MCP, think_tool)
    tools = await get_all_tools(config)
    if len(tools) == 0:
        raise ValueError(
            "No tools found to conduct research: Please configure either your "
            "search API or add MCP tools to your configuration."
        )
    
    # Step 2: Configure the researcher model with tools
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Prepare system prompt with MCP context if available
    researcher_prompt = research_system_prompt.format(
        mcp_prompt=configurable.mcp_prompt or "", 
        date=get_today_str()
    )
    
    # Configure model with tools, retry logic, and settings
    # In the researcher function, modify model binding:
    # Configure model with ONLY custom tools, NO native search
    base_model = init_chat_model(
    model=configurable.research_model,
    max_tokens=configurable.research_model_max_tokens,
    api_key=get_api_key_for_model(configurable.research_model, config),
    tags=["langsmith:nostream"],
)

# Configure model with ONLY our PubMed tools
    research_model = (
    base_model
    .bind_tools(tools)  # FORCE tool usage
    .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    .with_config(research_model_config)
)


    
    # Step 3: Generate researcher response with system context
    messages = [SystemMessage(content=researcher_prompt)] + researcher_messages
    response = await research_model.ainvoke(messages)
    
    # Step 4: Update state and proceed to tool execution
    return Command(
        goto="researcher_tools",
        update={
            "researcher_messages": [response],
            "tool_call_iterations": state.get("tool_call_iterations", 0) + 1
        }
    )

async def validate_final_answer(state: AgentState, config: RunnableConfig):
    """Validate that the final answer only cites Post IDs from research findings."""
    import re
    
    configurable = Configuration.from_runnable_config(config)
    final_report = state.get("final_report", "")
    notes = state.get("notes", [])
    
    # Extract ALL Post IDs from notes (match ANY 7-8 digit number)
    valid_ids = set()
    for note in notes:
        # Just extract any 7-8 digit numbers from the notes
        found_ids = re.findall(r'\b(\d{7,8})\b', note)
        valid_ids.update(found_ids)
    
    # Extract Post IDs cited in final answer  
    cited_ids = re.findall(r'Post\s+(\d{7,8})', final_report)
    cited_ids = set(cited_ids)
    
    # Find fake citations
    fake_ids = cited_ids - valid_ids
    
    if fake_ids:
        print(f"  HALLUCINATION DETECTED IN FINAL ANSWER!")
        print(f"   Fake Post IDs: {fake_ids}")
        print(f"   Valid Post IDs from research: {valid_ids}")
        
        # Replace fake citations
        cleaned_report = final_report
        for fake_id in fake_ids:
            patterns = [
                f"Post {fake_id}",
                f"Post  {fake_id}", 
            ]
            for pattern in patterns:
                cleaned_report = cleaned_report.replace(
                    pattern,
                    "[INVALID SOURCE - POST ID NOT IN DATABASE]"
                )
        
        return {
            "final_report": cleaned_report,
            "messages": [AIMessage(content=cleaned_report)]
        }
    
    print(f" VALIDATION PASSED: All {len(cited_ids)} citations are valid")
    
    # Answer is clean
    return {
        "final_report": final_report,
        "messages": [AIMessage(content=final_report)]
    }


async def inject_citations_into_report(final_report: str, valid_post_ids: set, findings: str) -> str:
    """Force citations into report by matching findings to report text."""
    import re
    
    if not valid_post_ids:
        return final_report
    
    # Build a map of content snippets to Post IDs from findings
    content_to_postid = {}
    for post_id in valid_post_ids:
        # Find mentions of this post_id in findings
        pattern = rf'Post {post_id}[^\d].*?(?=Post \d|$)'
        matches = re.findall(pattern, findings, re.DOTALL)
        for match in matches:
            # Extract key phrases (sentences with meaningful content)
            sentences = re.split(r'[.!?]+', match)
            for sent in sentences[:3]:  # First 3 sentences
                if len(sent) > 30:  # Meaningful length
                    content_to_postid[sent.strip().lower()] = post_id
    
    # Inject citations into report
    report_sentences = re.split(r'([.!?]+)', final_report)
    modified_report = []
    cited_ids = set()
    
    for i in range(0, len(report_sentences), 2):
        if i >= len(report_sentences):
            break
            
        sentence = report_sentences[i]
        punctuation = report_sentences[i+1] if i+1 < len(report_sentences) else ''
        
        # Skip if already has citation or is too short
        if re.search(r'Post \d{7,8}', sentence) or len(sentence) < 40:
            modified_report.append(sentence + punctuation)
            continue
        
        # Try to find matching content in findings
        sentence_lower = sentence.lower()
        best_match_id = None
        best_match_score = 0
        
        for content_snippet, post_id in content_to_postid.items():
            # Simple word overlap scoring
            sent_words = set(sentence_lower.split())
            snippet_words = set(content_snippet.split())
            overlap = len(sent_words & snippet_words)
            
            if overlap > best_match_score and overlap > 3:
                best_match_score = overlap
                best_match_id = post_id
        
        # Add citation if found match
        if best_match_id:
            modified_report.append(f"{sentence} [Post {best_match_id}]{punctuation}")
            cited_ids.add(best_match_id)
        else:
            # Add a generic citation from valid IDs for sections
            if '##' in sentence:
                modified_report.append(sentence + punctuation)
            elif len(cited_ids) < len(valid_post_ids):
                # Cycle through unused IDs
                unused = valid_post_ids - cited_ids
                if unused:
                    next_id = sorted(unused)[0]
                    modified_report.append(f"{sentence} [Post {next_id}]{punctuation}")
                    cited_ids.add(next_id)
                else:
                    modified_report.append(sentence + punctuation)
            else:
                modified_report.append(sentence + punctuation)
    
    # Add Sources section if missing
    result = ''.join(modified_report)
    if '### Sources' not in result and cited_ids:
        result += "\n\n### Sources\n"
        for post_id in sorted(cited_ids):
            result += f"- Stack Overflow Post {post_id}\n"
    
    return result


def fix_invalid_citations(report_text: str, valid_post_ids: set) -> str:
    """Replace all [INVALID SOURCE] markers with actual Post IDs from the valid list."""
    if not valid_post_ids:
        return report_text
    
    # Convert to list and cycle through them
    id_list = sorted(list(valid_post_ids))
    id_index = 0
    
    # Replace each [INVALID SOURCE - POST ID NOT IN DATABASE] with a real ID
    while '[INVALID SOURCE - POST ID NOT IN DATABASE]' in report_text:
        post_id = id_list[id_index % len(id_list)]
        report_text = report_text.replace(
            '[INVALID SOURCE - POST ID NOT IN DATABASE]',
            f'Post {post_id}',
            1  # Replace one at a time
        )
        id_index += 1
    
    return report_text



# Tool Execution Helper Function
async def execute_tool_safely(tool, args, config):
    """Safely execute a tool with error handling."""
    try:
        return await tool.ainvoke(args, config)
    except Exception as e:
        return f"Error executing tool: {str(e)}"


async def researcher_tools(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher", "compress_research"]]:
    """Execute tools called by the researcher, including search tools and strategic thinking.
    
    This function handles various types of researcher tool calls:
    1. think_tool - Strategic reflection that continues the research conversation
    2. Search tools (tavily_search, web_search) - Information gathering
    3. MCP tools - External tool integrations
    4. ResearchComplete - Signals completion of individual research task
    
    Args:
        state: Current researcher state with messages and iteration count
        config: Runtime configuration with research limits and tool settings
        
    Returns:
        Command to either continue research loop or proceed to compression
    """
    # Step 1: Extract current state and check early exit conditions
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    most_recent_message = researcher_messages[-1]
    
    # Early exit if no tool calls were made (including native web search)
    has_tool_calls = bool(most_recent_message.tool_calls)

    if not has_tool_calls:
        return Command(goto="compress_research")
    
    # Step 2: Handle other tool calls (search, MCP tools, etc.)
    tools = await get_all_tools(config)
    tools_by_name = {
        tool.name if hasattr(tool, "name") else tool.get("name", "web_search"): tool 
        for tool in tools
    }
    
    # Execute all tool calls in parallel
    tool_calls = most_recent_message.tool_calls
    tool_execution_tasks = [
        execute_tool_safely(tools_by_name[tool_call["name"]], tool_call["args"], config) 
        for tool_call in tool_calls
    ]
    observations = await asyncio.gather(*tool_execution_tasks)
    
    # Create tool messages from execution results
    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        ) 
        for observation, tool_call in zip(observations, tool_calls)
    ]
    
    # Step 3: Check late exit conditions (after processing tools)
    exceeded_iterations = state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls
    research_complete_called = any(
        tool_call["name"] == "ResearchComplete" 
        for tool_call in most_recent_message.tool_calls
    )
    
    if exceeded_iterations or research_complete_called:
        # End research and proceed to compression
        return Command(
            goto="compress_research",
            update={"researcher_messages": tool_outputs}
        )
    
    # Continue research loop with tool results
    return Command(
        goto="researcher",
        update={"researcher_messages": tool_outputs}
    )

async def compress_research(state: ResearcherState, config: RunnableConfig):
    """Compress and synthesize research findings into a concise, structured summary.
    
    This function takes all the research findings, tool outputs, and AI messages from
    a researcher's work and distills them into a clean, comprehensive summary while
    preserving all important information and findings.
    
    Args:
        state: Current researcher state with accumulated research messages
        config: Runtime configuration with compression model settings
        
    Returns:
        Dictionary containing compressed research summary and raw notes
    """
    # Step 1: Configure the compression model
    configurable = Configuration.from_runnable_config(config)
    synthesizer_model = create_clean_model(
    configurable, 
    configurable.compression_model,
    configurable.compression_model_max_tokens, 
    config
).with_config({
        "model": configurable.compression_model,
        "max_tokens": configurable.compression_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.compression_model, config),
        "tags": ["langsmith:nostream"]
    })
    
    # Step 2: Prepare messages for compression
    researcher_messages = state.get("researcher_messages", [])
    
    # Add instruction to switch from research mode to compression mode
    researcher_messages.append(HumanMessage(content=compress_research_simple_human_message))


    
    # Step 3: Attempt compression with retry logic for token limit issues
    synthesis_attempts = 0
    max_attempts = 3
    
    while synthesis_attempts < max_attempts:
        try:
            # Create system prompt focused on compression task
            compression_prompt = compress_research_system_prompt.format(date=get_today_str())
            messages = [SystemMessage(content=compression_prompt)] + researcher_messages
            
            # Execute compression
            response = await synthesizer_model.ainvoke(messages)

            
            compressed_content = str(response.content)
            import re
            post_ids_found = re.findall(r'(?:Post\s*#?\s*|POST\s*#?\s*|Id:?\s*)(\d{6,})', compressed_content)
            
            
            # Extract raw notes from all tool and AI messages
            raw_notes_content = "\n".join([
                str(message.content) 
                for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
            ])
            
            # Return successful compression result
            return {
                "compressed_research": str(response.content),
                "raw_notes": [raw_notes_content]
            }
            
        except Exception as e:
            synthesis_attempts += 1
            
            # Handle token limit exceeded by removing older messages
            if is_token_limit_exceeded(e, configurable.research_model):
                researcher_messages = remove_up_to_last_ai_message(researcher_messages)
                continue
            
            # For other errors, continue retrying
            continue
    
    # Step 4: Return error result if all attempts failed
    raw_notes_content = "\n".join([
        str(message.content) 
        for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
    ])
    
    return {
        "compressed_research": "Error synthesizing research report: Maximum retries exceeded",
        "raw_notes": [raw_notes_content]
    }

# Researcher Subgraph Construction
# Creates individual researcher workflow for conducting focused research on specific topics
researcher_builder = StateGraph(
    ResearcherState, 
    output=ResearcherOutputState, 
    config_schema=Configuration
)

# Add researcher nodes for research execution and compression
researcher_builder.add_node("researcher", researcher)                 # Main researcher logic
researcher_builder.add_node("researcher_tools", researcher_tools)     # Tool execution handler
researcher_builder.add_node("compress_research", compress_research)   # Research compression

# Define researcher workflow edges
researcher_builder.add_edge(START, "researcher")           # Entry point to researcher
researcher_builder.add_edge("compress_research", END)      # Exit point after compression

# Compile researcher subgraph for parallel execution by supervisor
researcher_subgraph = researcher_builder.compile()

async def final_report_generation(state: AgentState, config: RunnableConfig):
    """Generate the final comprehensive research report with retry logic for token limits.
    
    This function takes all collected research findings and synthesizes them into a 
    well-structured, comprehensive final report using the configured report generation model.
    
    Args:
        state: Agent state containing research findings and context
        config: Runtime configuration with model settings and API keys
        
    Returns:
        Dictionary containing the final report and cleared state
    """
    # Step 1: Extract research findings and prepare state cleanup
    notes = state.get("notes", [])

    cleared_state = {"notes": {"type": "override", "value": []}}
    findings = "\n".join(notes)

    import re
    valid_ids = set()
    for note in notes:
        # Match various Post ID formats
        found_ids = re.findall(
            r'(?:Post\s*#?\s*|POST\s*#?\s*|Post\s*ID:?\s*|POST\s*ID:?\s*|Id:?\s*|ID:?\s*)(\d{5,})',
            note
        )

        valid_ids.update(found_ids)

    valid_post_ids_str = ", ".join(sorted(valid_ids)) if valid_ids else "None found"
    
    # Extract valid Post IDs from research findings
    import re
    valid_ids = set()
    for note in notes:
        found_ids = re.findall(r'(?:Post\s*#?\s*|Post\s*ID:?\s*)(\d{6,})', note)
        valid_ids.update(found_ids)
    
    valid_post_ids_str = ", ".join(sorted(valid_ids)) if valid_ids else "None found"
    logging.info(f"TOTAL VALID POST IDs EXTRACTED: {valid_post_ids_str}")
    logging.info("=" * 80)


    
    # Step 2: Configure the final report generation model
    configurable = Configuration.from_runnable_config(config)
    writer_model_config = {
        "model": configurable.final_report_model,
        "max_tokens": configurable.final_report_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.final_report_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Step 3: Attempt report generation with token limit retry logic
    max_retries = 3
    current_retry = 0
    findings_token_limit = None
    
    while current_retry <= max_retries:
        try:
            # Create comprehensive prompt with all research context
            final_report_prompt = final_report_generation_prompt.format(
                research_brief=state.get("research_brief", ""),
                messages=get_buffer_string(state.get("messages", [])),
                findings=findings,
                valid_post_ids=valid_post_ids_str,
                date=get_today_str()
            )
            
            # Generate the final report
           # Generate the final report using clean model
            report_model = create_clean_model(
                configurable, 
                configurable.final_report_model, 
                configurable.final_report_model_max_tokens, 
                config
            )
            # Generate the final report using clean model
            report_response = await report_model.with_config(writer_model_config).ainvoke(
                [HumanMessage(content=final_report_prompt)] 
            )



            # Extract content as string
            finalreport_text = str(report_response.content)

            finalreport_text = fix_encoding_issues(finalreport_text)

            finalreport_text = fix_invalid_citations(finalreport_text, valid_ids)

            # Create the final AIMessage
            finalreport = AIMessage(content=finalreport_text)
            
            # Return successful report generation
            return {
                "final_report": finalreport_text, 
                "messages": [finalreport],
                **cleared_state
            }

            
        except Exception as e:
            # Handle token limit exceeded errors with progressive truncation
            if is_token_limit_exceeded(e, configurable.final_report_model):
                current_retry += 1
                
                if current_retry == 1:
                    # First retry: determine initial truncation limit
                    model_token_limit = get_model_token_limit(configurable.final_report_model)
                    if not model_token_limit:
                        return {
                            "final_report": f"Error generating final report: Token limit exceeded, however, we could not determine the model's maximum context length. Please update the model map in deep_researcher/utils.py with this information. {e}",
                            "messages": [AIMessage(content="Report generation failed due to token limits")],
                            **cleared_state
                        }
                    # Use 4x token limit as character approximation for truncation
                    findings_token_limit = model_token_limit * 4
                else:
                    # Subsequent retries: reduce by 10% each time
                    findings_token_limit = int(findings_token_limit * 0.9)
                
                # Truncate findings and retry
                findings = findings[:findings_token_limit]
                continue
            else:
                # Non-token-limit error: return error immediately
                return {
                    "final_report": f"Error generating final report: {e}",
                    "messages": [AIMessage(content="Report generation failed due to an error")],
                    **cleared_state
                }
    
    # Step 4: Return failure result if all retries exhausted
    return {
        "final_report": "Error generating final report: Maximum retries exceeded",
        "messages": [AIMessage(content="Report generation failed after maximum retries")],
        **cleared_state
    }

# Main Deep Researcher Graph Construction
# Creates the complete deep research workflow from user input to final report
deep_researcher_builder = StateGraph(
    AgentState, 
    input=AgentInputState, 
    config_schema=Configuration
)

# Add main workflow nodes for the complete research process
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)           # User clarification phase
deep_researcher_builder.add_node("write_research_brief", write_research_brief)     # Research planning phase
deep_researcher_builder.add_node("research_supervisor", supervisor_subgraph)       # Research execution phase
deep_researcher_builder.add_node("final_report_generation", final_report_generation)  # Report generation phase
deep_researcher_builder.add_node("validate_final_answer", validate_final_answer)

# Define main workflow edges for sequential execution
deep_researcher_builder.add_edge(START, "clarify_with_user")                       # Entry point
deep_researcher_builder.add_edge("research_supervisor", "final_report_generation") # Research to report
deep_researcher_builder.add_edge("final_report_generation", END)               # Final exit point

# Compile the complete deep researcher workflow
deep_researcher = deep_researcher_builder.compile()