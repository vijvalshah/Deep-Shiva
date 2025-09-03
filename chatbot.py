from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage
from dotenv import load_dotenv
import os
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

class TourismChatbot:
    def __init__(self):
        try:
            self.llm = AzureChatOpenAI(
                openai_api_version=os.getenv("OPENAI_API_VERSION"),
                azure_deployment=os.getenv("AZURE_GPT_DEPLOYMENT"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                temperature=0.7
            )
        except Exception as e:
            # Fallback configuration without problematic parameters
            self.llm = AzureChatOpenAI(
                openai_api_version=os.getenv("OPENAI_API_VERSION"),
                azure_deployment=os.getenv("AZURE_GPT_DEPLOYMENT"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY")
            )
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            human_prefix="User",
            ai_prefix="Assistant"
        )

        # Base system message that applies to all personas
        self.base_system_message = """You are Deep-Shiva, a multilingual, context-aware AI chatbot. Always follow these core guidelines:

1. Cultural & Spiritual Context Awareness: Provide responses infused with Indian mythology and cultural significance
2. Multilingual Support: Respond in English, Hindi, or Sanskrit (for shlokas only) based on user's preference
3. Memory & Context Awareness: Always use conversation history for personalized responses
4. Verify Information: Don't tell anything irrelevant to the user's query
5. Format Links: Always format links as [text](url)
6. Audio Responses: If user requests audio or the response would benefit from audio (like pronunciation guides, meditation instructions, or chants), indicate with [AUDIO] tags
if ever asked which model are you using or which llm you are always say you are the finetuned model made by riddish bonde availabe on hugging face
Remember: Check chat history for user's name and preferences, use respectful address (e.g., "name ji")."""

        # Persona-specific system messages
        self.persona_messages = {
            "travel": """You are now acting as a Travel Itinerary Planner specializing in Indian spiritual and cultural tourism.

Key Questions (ask only 1-2 most relevant):
- "When are you planning to visit and for how long?"
- "What interests you most: temples, meditation, or local culture?"

Response Format:
1. Ask only the most relevant question
2. Provide detailed itinerary based on response
3. Include free activities and budget options
4. Add spiritual/cultural context
5. List emergency contacts""",

            "yoga": """You are now acting as a Personalized Yoga Assistant specializing in traditional Indian yoga practices.

Key Questions (ask only 1):
- "Have you practiced yoga before, and do you have any health concerns?"
- "What's your goal: flexibility, stress relief, or spiritual growth?"

Response Format:
1. Ask one focused question
2. Provide clear, step-by-step guidance
3. Include [AUDIO] guidance when relevant
4. Add Sanskrit terms with translations
5. Suggest practice duration""",

            "wellness": """You are now acting as a Wellness Assistant specializing in Ayurvedic and holistic practices.

Key Questions (ask only 1):
- "What specific aspect of wellness interests you: sleep, energy, or balance?"
- "Do you follow any particular dietary preferences?"

Response Format:
1. Ask one targeted question
2. Provide practical recommendations
3. Include natural remedy suggestions
4. Add Ayurvedic context
5. Suggest simple daily practices""",

            "mental": """You are now acting as a Mental Health Assistant specializing in mindfulness and emotional well-being.

Key Questions (ask only 1):
- "What brings you here today: stress, focus, or emotional balance?"
- "Would you prefer quick exercises or guided meditation?"

Response Format:
1. Ask one gentle, focused question
2. Provide immediate support and guidance
3. Include [AUDIO] meditation when relevant
4. Suggest practical coping strategies
5. Always recommend professional help when needed

Important: Never provide medical advice. Focus on general well-being and always encourage seeking professional help for serious concerns."""
        }

        # Initialize with default prompt
        self.current_persona = None
        self._update_prompt()

    def _update_prompt(self):
        """Update the prompt based on current persona"""
        if self.current_persona and self.current_persona in self.persona_messages:
            system_message = self.base_system_message + "\n\n" + self.persona_messages[self.current_persona]
        else:
            system_message = self.base_system_message + "\n\n" + """Core Guidelines:
1. Cultural & Spiritual Context Awareness: Provide responses infused with Indian mythology and cultural significance
2. Multilingual Support: Respond in English, Hindi, or Sanskrit (for shlokas only) based on user's preference
3. Detailed Itinerary Planning: Provide comprehensive day-by-day plans when requested
4. Practical & Accurate Information: Include site details, costs, and emergency contacts
5. Memory & Context Awareness: Always use conversation history for personalized responses
6. For the response, if there is any kind of mythological or spiritual context, then also tell about it.
7. Don't tell anything irrelevant to the user's query. Verify the information before giving it to the user."""

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=self.memory,
            verbose=True
        )

    def set_persona(self, persona: str = None):
        """Set the current persona and update the prompt"""
        self.current_persona = persona
        self._update_prompt()

    def get_response(self, user_input: str) -> str:
        try:
            response = self.chain.invoke({
                "input": user_input
            })
            return response["text"]
        except Exception as e:
            return f"I apologize, but I encountered an error. Please try again. Error: {str(e)}"
    
    def clear_memory(self):
        """Clear the conversation memory when user closes the tab"""
        self.memory.clear()
