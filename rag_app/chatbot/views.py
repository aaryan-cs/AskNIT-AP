from django.shortcuts import render

# Create your views here.
from django.http import JsonResponse
from .utils.rag_engine import get_bot_response

def index(request):
    return render(request, 'chatbot/index.html')

def chat_api(request):
    if request.method == "POST":
        user_input = request.POST.get("message", "")
        print("User asked:", user_input)  
        response = get_bot_response(user_input)
        print("Bot responded:", response) 
        return JsonResponse({"response": response})
