"""Task 1: Simple Rule-Based Chatbot
A chatbot that responds to user inputs using pattern matching and predefined rules.
"""

import re

class RuleBasedChatbot:
    def __init__(self):
        self.responses = {
            r'(hi|hello|hey)': 'Hello! How can I help you today?',
            r'(what is your name|who are you)': 'I am a simple chatbot created for CODSOFT.',
            r'(how are you|how\'s it going)': 'I\'m doing great! Thanks for asking.',
            r'(what time is it|current time)': 'I don\'t have access to the current time, but you can check your device.',
            r'(tell me a joke)': 'Why did the AI go to school? To improve its neural network!',
            r'(bye|goodbye|see you)': 'Goodbye! Have a great day!',
            r'(thank you|thanks)': 'You\'re welcome! Happy to help.',
            r'(help|what can you do)': 'I can chat with you, tell jokes, and answer basic questions.',
        }
    
    def get_response(self, user_input):
        """Generate a response based on user input"""
        user_input = user_input.lower()
        
        for pattern, response in self.responses.items():
            if re.search(pattern, user_input):
                return response
        
        return "I didn't understand that. Can you rephrase?"
    
    def chat(self):
        """Main chat loop"""
        print("\n=== Welcome to the Rule-Based Chatbot ===")
        print("Type 'bye' to exit the chatbot.\n")
        
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            
            response = self.get_response(user_input)
            print(f"Bot: {response}\n")
            
            if re.search(r'(bye|goodbye|exit)', user_input.lower()):
                break

if __name__ == "__main__":
    chatbot = RuleBasedChatbot()
    chatbot.chat()
