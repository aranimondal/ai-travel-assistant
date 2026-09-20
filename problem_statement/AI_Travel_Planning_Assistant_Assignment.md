# AI TRAVEL PLANNING ASSISTANT

Developer Assignment Brief

Build a context-aware travel assistant that combines a document-based knowledge base with current information retrieved through MCP tools.

## 1. Background

Travel planning requires two distinct types of information: destination knowledge, such as attractions, neighbourhoods, transportation options, cultural guidance, and suggested itineraries; and current information, such as weather forecasts and currency exchange rates.

Destination knowledge can be collected in advance and searched using Retrieval-Augmented Generation. Current information should be requested from external services when the user asks a time-sensitive question.

## 2. Problem Statement

Build an AI Travel Planning Assistant that helps users plan a trip to a selected destination. The assistant must use a travel knowledge base to provide destination recommendations and use MCP tools to retrieve current information. It should combine both sources when a question requires destination knowledge as well as current information.

Example: “Plan a three-day trip to Singapore and adjust the activities based on the weather forecast.”

For this request, the application should retrieve attractions and itinerary recommendations from the knowledge base, invoke an MCP weather tool, generate a weather-aware itinerary, and identify the sources and tool results used.

## 3. Application Scope

Candidates should implement the solution for one destination. Singapore is recommended because suitable public travel resources are readily available. Another destination may be selected if the knowledge base contains sufficient information about attractions, transportation, cultural guidance, and itinerary planning.

The application does not need to support flight or hotel booking, payment processing, route navigation, or travel reservations.

## 4. Core Features

### 4.1 Destination Knowledge Assistant using RAG

Create a document-based knowledge base for the selected destination. It should cover:

- Major attractions and neighbourhoods
- Local transportation guidance
- Cultural and practical travel tips
- Food and local experiences
- Sample itineraries
- Indoor and outdoor activity suggestions

The application should answer destination questions such as:

- What are the must-visit attractions in Singapore?
- Which neighbourhoods are suitable for cultural experiences?
- How can a tourist travel around Singapore?
- Suggest activities for a family with children.
- Create a three-day sightseeing itinerary.
- What indoor attractions can I visit?

#### RAG requirements

1. Load travel content from public documents or web pages.
2. Divide content into meaningful chunks.
3. Generate embeddings for the chunks.
4. Store embeddings in a vector store.
5. Retrieve relevant chunks for each user question.
6. Generate answers grounded in the retrieved content.
7. Display the source title or source link used for the answer.

If the knowledge base does not contain enough information, the assistant must state this clearly instead of inventing destination facts.

### 4.2 Current Travel Information using MCP

Use MCP tools for current information obtained from external services. The solution must integrate the following two capabilities.

#### MCP Tool 1: Weather information

Retrieve current conditions or a forecast for the selected destination. Example questions include:

- What is the weather in Singapore?
- What is the forecast for the next three days?
- Is rain expected during my trip?
- Should I plan indoor or outdoor activities tomorrow?

#### MCP Tool 2: Currency conversion

Convert an amount between two currencies using information returned by the connected service. Example questions include:

- Convert INR 50,000 to SGD.
- How much is 200 SGD in INR?
- Convert my travel budget from USD to Singapore dollars.

#### MCP requirements

8. Connect to at least two MCP tools.
9. Make the tools available to the AI application.
10. Select the appropriate tool based on the user request.
11. Pass the required input to the selected tool.
12. Use the returned information in the final response.
13. Clearly indicate when current information came from an MCP tool.
14. Handle unavailable tools or failed results without fabricating an answer.

MCP should not be used to answer destination questions already covered by the knowledge base.

### 4.3 Combined RAG and MCP Response

The primary feature is a response that combines stable destination knowledge with current information.

Required combined scenario: Create a three-day Singapore itinerary for next week and adjust it according to the weather forecast.

The application should use RAG to retrieve relevant attractions, indoor and outdoor activities, itinerary suggestions, and transportation guidance. It should use the MCP weather tool for the forecast, then produce a day-wise, weather-aware itinerary with indoor alternatives where appropriate.

Additional combined scenarios may include:

- I have a budget of INR 60,000. Convert it to SGD and suggest a three-day itinerary.
- Suggest outdoor attractions and replace them with indoor options if rain is expected.
- Plan a family trip and include the latest weather forecast.
- Create a cultural itinerary and show my budget in the destination currency.

The response should distinguish knowledge-base facts, MCP-provided current information, and recommendations generated by the LLM.

## 5. Prompt Engineering Requirements

Design prompts that instruct the model to:

- Use retrieved knowledge-base content for destination facts.
- Use MCP responses for current information.
- Avoid presenting unsupported information as fact.
- State when sufficient information is unavailable.
- Produce clear, structured travel recommendations.
- Include source references where applicable.
- Distinguish factual information from AI-generated suggestions.
- Preserve relevant user preferences from the conversation.

Briefly explain the prompt strategy in the project documentation.

## 6. Suggested Knowledge Base

Use at least three public resources for one destination. The following Singapore resources are recommended:

- Wikivoyage Singapore Travel Guide — Districts, attractions, transportation, food, practical guidance, and itineraries.
- Visit Singapore: Essential Travel Information — Practical visitor information, climate, language, connectivity, and useful services.
- Visit Singapore: Sample Itineraries — Itinerary ideas for different durations and traveller profiles.
- Visit Singapore: Things to Do — Attractions and activities across multiple visitor-interest categories.

Candidates may convert selected content into PDF, Markdown, HTML, or plain-text files for ingestion. The original source title and URL should be retained as metadata so the application can show meaningful citations. Review and follow each source’s reuse terms when redistributing extracted content.

## 7. Technology Requirements

- LangChain for application orchestration, prompts, retrieval, and tool integration
- An appropriate embedding model
- A vector store such as FAISS or Chroma
- An LLM of the candidate’s choice
- An MCP-compatible client and MCP tools
- A basic user interface using tech/framework of choice.

The focus is the AI workflow and integration quality rather than sophisticated interface design.

## 8. Minimum Acceptance Criteria

- Knowledge base created from at least three travel resources
- Embedding-based semantic retrieval
- Grounded answers with source references
- Weather information through an MCP tool
- Currency conversion through an MCP tool
- At least one response combining RAG and MCP
- Multi-turn conversation with retained context
- Appropriate tool selection based on user intent
- Clear handling of missing knowledge and tool failures
- A simple, usable interface

## 9. Deliverables

15. Source code in a Git repository
16. A working application
17. Knowledge-base documents, or clear instructions for obtaining them
18. A README covering architecture, knowledge-base sources, RAG workflow, MCP tools, prompt and context strategy, and setup instructions
19. Sample questions and application responses
20. A short demonstration showing RAG, MCP, a combined response, and conversational context
