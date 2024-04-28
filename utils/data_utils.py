
def make_prompt(context, answer, question=None):
    instruction = "### Hãy tạo CÂU HỎI trên ĐOẠN VĂN và CÂU TRẢ LỜI sau:"
    prompt = instruction + "\n### ĐOẠN VĂN: " + context + "\n### CÂU TRẢ LỜI: " + answer + "\n### CÂU HỎI: "
    prompt += question if question else ""
    return prompt