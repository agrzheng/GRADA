MULTIPLE_PROMPT = 'You are a helpful assistant, below is a query from a user and some relevant contexts. \
Answer the question given the information in those contexts. Your answer should be short ,concise and without explainations ,just answer with one or two words like yes or no. \
If you cannot find the answer to the question, just say "I don\'t know". \
\n\nContexts: [context] \n\nQuery: [question] \n\nAnswer:'

def wrap_prompt(question, context, prompt_id=1) -> str:
    assert type(context) == list
    context_str = "\n".join(context)
    
    input_prompt = MULTIPLE_PROMPT.replace('[question]', question).replace('[context]', context_str)

    return input_prompt

