CHAT_TEMPLATE = {
    "vistral" : """[INST] <<SYS>>
Bạn là một trợ lí Tiếng Việt nhiệt tình và trung thực. Hãy luôn trả lời một cách hữu ích nhất có thể, đồng thời giữ an toàn.
Câu trả lời của bạn không nên chứa bất kỳ nội dung gây hại, phân biệt chủng tộc, phân biệt giới tính, độc hại, nguy hiểm hoặc bất hợp pháp nào. Hãy đảm bảo rằng các câu trả lời của bạn không có thiên kiến xã hội và mang tính tích cực.Nếu một câu hỏi không có ý nghĩa hoặc không hợp lý về mặt thông tin, hãy giải thích tại sao thay vì trả lời một điều gì đó không chính xác. Nếu bạn không biết câu trả lời cho một câu hỏi, hãy trả lời là bạn không biết và vui lòng không chia sẻ thông tin sai lệch.
<</SYS>>

{} [/INST]""",
    "space" : "{} "
}

def make_prompt(context, answer, question=None, template=None):
    instruction = "### Hãy tạo Câu hỏi dựa trên Đoạn văn và Câu trả lời sau:"
    prompt = instruction + "\n### Đoạn văn: " + context + "\n### Câu trả lời: " + answer + "\n### Câu hỏi:"
    if template:
        prompt = CHAT_TEMPLATE[template].format(prompt)
    prompt += question if question else ""

    return prompt

def make_reward_input(context, answer, question):
    text =  context + " ### " + answer + " ### " + question
    
    return text