import google.generativeai as genai
import os

# 1. Setup API Key (Replace with your actual Gemini API Key)
# In production, use os.getenv("GEMINI_API_KEY") with a .env file
GEMINI_API_KEY = "AQ.Ab8RN6IzQewMoh49U-iUymBBPyNBy54loJIuLl7AF_34zDPFag"
genai.configure(api_key=GEMINI_API_KEY, transport="rest")
# 2. Define the System Prompt for the AI Agronomist
system_instruction = """
You are 'Krishi Expert AI', an advanced agricultural AI assistant designed to help farmers in Palwal, Haryana.
Your primary role is to provide actionable, accurate, and safe treatment advice for crop diseases (like Yellow Rust, Aphids, etc.).
You must respond in the language the user speaks (Hindi, Haryanvi, or English). 
Keep your answers brief, practical, and highly respectful (use terms like 'Kisan bhai' or 'ji').
If recommending chemicals, always specify the dosage and safety precautions (e.g., wearing a mask).
"""

# 3. Initialize the Model
model = genai.GenerativeModel(
    model_name="gemini-3.8-flash",
    system_instruction=system_instruction
)

# 4. Start an Interactive Chat Session
chat = model.start_chat(history=[])

print("🌾 Krishi Expert AI is online! Type 'exit' to quit.")
print("-" * 50)

while True:
    # .strip() removes accidental spaces
    user_input = input("👨‍🌾 You: ").strip() 
    
    if user_input.lower() == 'exit':
        print("Closing chat. Good luck with the hackathon!")
        break
        
    # Prevent empty messages from crashing the app
    if not user_input:
        print("⚠️ Please type a message before pressing Enter.")
        continue
        
    # Send the message to Gemini
    response = chat.send_message(user_input)
    
    print(f"\n🤖 AI Expert: {response.text}")
    print("-" * 50)