#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

import openai

# Replace with your actual API key or set it as an environment variable
api_key = "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"  # Replace this with your new API key

try:
    result = 10 / 0  # This will raise a ZeroDivisionError
except Exception as error:
    client = openai.OpenAI(api_key=api_key)  # Pass API key here
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": f"Fix for: {str(error)}"}]
    )
    fix = response.choices[0].message.content
    print(fix)

