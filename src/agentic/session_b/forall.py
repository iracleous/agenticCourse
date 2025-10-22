def getAnswer(prompt: str) -> str:
    """
    Simulates getting an answer from a language model based on the provided prompt.
    In a real implementation, this function would interface with an LLM API.
    """
    # For demonstration purposes, we'll return a simple echoed response.
    return f"Echo: {prompt}"

yesContinue = True
while yesContinue:
    prompt = input("Enter your prompt: ")
    response = getAnswer(prompt)
    print("Response:", response)

    # Get user input 
    user_input = input("Do you want to continue? (yes/ anything else terminates): ").strip().lower()
    if user_input == 'yes':
        print("Continuing...")
    else:
        print("Exiting...")
        yesContinue = False


