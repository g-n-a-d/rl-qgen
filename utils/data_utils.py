def make_prompt(context, answer):
    instruction = "### Hãy tạo CÂU HỎI dựa trên ĐOẠN VĂN và CÂU TRẢ LỜI sau:"
    prompt = instruction + "\n### ĐOẠN VĂN: " + context + "\n### CÂU TRẢ LỜI: " + answer + "\n### CÂU HỎI: "
    
    return prompt

def make_reward_input(context, answer, question):
    text =  context + " ### " + answer + " ### " + question
    
    return text