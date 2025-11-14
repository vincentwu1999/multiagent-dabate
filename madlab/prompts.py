PROMPTS = {
    "arithmetic_start": "What is the result of {}+{}*{}+{}-{}*{}? Make sure to state your answer at the end of the response.",
    "arithmetic_debate": (
        "These are the recent/updated opinions from other agents:\n{other}\n"
        "Use these opinions carefully as additional advice, can you provide an updated answer? "
        "Make sure to state your answer at the end of the response."
        "At the very end, add a line: Confidence: XX%."),
    ),
    "gsm8k_start": (
        "Can you solve the following math problem?\n{problem}\n"
        "Explain your reasoning. Your final answer should be a single numerical number, "
        "in the form \\boxed{{answer}}, at the end of your response."
        "At the very end, add a line: Confidence: XX%." # This line is added for weighted voting
    ),
    "gsm8k_debate": (
        "These are the solutions to the problem from other agents:\n{other}\n"
        "Using the solutions from other agents as additional information, can you provide your answer to the math problem?\n"
        "The original math problem is:\n{problem}\n"
        "Your final answer should be a single numerical number, in the form \\boxed{{answer}}, at the end of your response."
        "At the very end, add a line: Confidence: XX%."
        
    ),
    "mmlu_start": (
        "Can you answer the following question as accurately as possible?\n\n{question}\n"
        "A) {A}\nB) {B}\nC) {C}\nD) {D}\n\n"
        "Explain your answer, putting the answer in the form (X) at the end of your response."
        "At the very end, add a line: Confidence: XX%."
    ),
    "mmlu_debate": (
        "These are the solutions to the problem from other agents:\n{other}\n"
        "Using the reasoning from other agents as additional advice, can you give an updated answer?\n"
        "Examine your solution and that of other agents. Put your answer in the form (X) at the end of your response."
        "At the very end, add a line: Confidence: XX%."
    ),
    "bio_start": (
        "Give a bullet point biography of {person} highlighting their contributions and achievements as a computer scientist, "
        "with each fact separated with a new line character."
        "At the very end, add a line: Confidence: XX%."
    ),
    "bio_debate": (
        "Here are some bullet point biographies of {person} given by other agents:\n{other}\n"
        "Closely examine your biography and the biography of other agents and provide an updated bullet point biography."
        "At the very end, add a line: Confidence: XX%."
    ),
    "chess_valid_start": (
        "Given the chess game\n{moves}\n"
        "give one valid destination square for the chess piece at {origin}.\n"
        "State the destination square in the form (X), where X follows the regex [a-h][1-8], for example (e5).\n"
        "Give a one line explanation of why your destination square is a valid move."
        "At the very end, add a line: Confidence: XX%."
    ),
    "chess_valid_debate": (
        "Here are destination square suggestions from other agents:\n{other}\n"
        "Can you double check that your destination square is a valid move? "
        "Check the valid move justifications from other agents.\n"
        "State your final answer in a newline with a 2 letter response following the regex [a-h][1-8]."
        "At the very end, add a line: Confidence: XX%."
    ),
    "bioasq_start": (
        "You are answering a biomedical question. Give a concise, factual answer.\n"
        "Question: {question}\n"
        "Type: {qtype}  (one of factoid | yesno | list)\n\n"
        "Rules:\n"
        "- If type is 'factoid', return a short entity/string.\n"
        "- If type is 'yesno', return only 'yes' or 'no'.\n"
        "- If type is 'list', return a comma-separated list of items.\n\n"
        "On the LAST line, write: FINAL: <your answer>\n"
        "At the very end, add a line: Confidence: XX%."
    ),
    "bioasq_debate": (
        "Here are other agents' answers:\n{other}\n\n"
        "Re-evaluate the question and provide your final answer following the rules.\n"
        "On the LAST line, write: FINAL: <your answer>\n"
        "At the very end, add a line: Confidence: XX%."
    ),
}
