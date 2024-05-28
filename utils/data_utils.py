def make_prompt(context, answer, question=None, last_space=True):
    instruction = "### Hãy tạo Câu hỏi dựa trên Đoạn văn và Câu trả lời sau:"
    prompt = instruction + "\n### Đoạn văn: " + context + "\n### Câu trả lời: " + answer + "\n### Câu hỏi:"
    prompt += " " if last_space else ""
    prompt += question if question else ""
    
    return prompt

def make_reward_input(context, answer, question):
    text =  context + " ### " + answer + " ### " + question
    
    return text