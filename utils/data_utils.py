def make_prompt(context, answer, question=None):
    instruction = "### Hãy tạo CÂU HỎI dựa trên ĐOẠN VĂN và CÂU TRẢ LỜI sau:"
    prompt = instruction + "\n### ĐOẠN VĂN: " + context + "\n### CÂU TRẢ LỜI: " + answer + "\n### CÂU HỎI: "
    prompt += question if question else ""
    
    return prompt

def make_prompt_(context, answer, question):
    instruction = "### Hãy tạo Câu hỏi dựa trên Đoạn văn và Câu trả lời sau:"
    prompt = instruction + "\n### Đoạn văn: " + context + "\n### Câu trả lời: " + answer + "\n### Câu hỏi: " + question
    
    return prompt

def make_reward_input(context, answer, question):
    text =  context + " ### " + answer + " ### " + question
    
    return text