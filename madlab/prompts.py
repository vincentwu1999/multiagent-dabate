PROMPTS = {
    "arithmetic_start": "What is the result of {}+{}*{}+{}-{}*{}? Make sure to state your answer at the end of the response.",
    "arithmetic_debate": (
        "These are the recent/updated opinions from other agents:\n{other}\n"
        "Use these opinions carefully as additional advice, can you provide an updated answer? "
        "Make sure to state your answer at the end of the response."
        "At the very end, add a line: Confidence: XX%."
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
    "chess_move_start": (
        "Here is the current sequence of moves in a chess game:\n{moves}\n"
        "What is the best chess move I should execute next? Give a single move suggestion of the form 14. <XXX> "
        "and make sure the chess move is valid in the current board state."
        "At the very end, add a line: Confidence: XX%."
    ),
    "chess_move_debate": (
        "Here are other chess move suggestions from other agents:\n{other}\n"
        "Using the chess suggestions from other agents as additional advice and your earlier generated solution, "
        "can you give me your updated thoughts on the best next chess move I should play given the chess sequence?\n"
        "Give a single move suggestion of the form 14. <XXX> and make sure the chess move is valid in the current board state."
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
    "debate_full_history_summary": (
        "Summarize this multi-agent debate (max 2500 chars). Use clustered formatting to avoid repeating identical answers.\n\n"
        "ROUND {round_num} RESPONSES:\n{responses}\n\n"
        "{previous_summary_section}"
        "OUTPUT (follow exactly):\n"
        "1. CONSENSUS SNAPSHOT:\n"
        "   - List each unique answer cluster as Answer=[X] | Agents=[IDs] | Confidence=[min–max%] | Key Reasoning=[≤150 chars].\n"
        "2. DIVERGENT OPINIONS:\n"
        "   - Only answers not in consensus clusters. Format Agent[ID]: Answer=[Y] | Confidence=[%] | Key Reasoning=[≤120 chars].\n"
        "3. CHANGE LOG:\n"
        "   - Include only agents whose answer changed or whose confidence shifted ≥10% vs previous round. "
        "Format Agent[ID]: was [old answer @ old conf], now [new answer @ new conf], rationale shift=[≤100 chars]. "
        "Write \"None\" if no changes.\n"
        "4. OPEN QUESTIONS:\n"
        "   - ≤150 chars describing unresolved disagreements or next steps needed for alignment.\n"
        "\nGuidelines: Preserve essential info from prior summary, integrate new reasoning, prioritize answers > confidence > reasoning, "
        "and stay extremely concise."
    ),
    # Full-history specific prompts (use {summary} placeholder instead of {other})
    # These prompts should produce responses in the same format as start_prompt (reasoning, answer, confidence)
    "arithmetic_full_history": (
        "Here is a condensed summary of all debate rounds so far:\n{summary}\n\n"
        "Review the debate summary to understand what answers and reasoning have been discussed. "
        "Use this summary to possibly update your answer. Re-evaluate your calculation step-by-step, incorporating any valid insights from the summary.\n"
        "Original problem: What is the result of {}+{}*{}+{}-{}*{}?\n"
        "Make sure to state your answer at the end of the response.\n"
        "At the very end, add a line: Confidence: XX%."
    ),
    "gsm8k_full_history": (
        "Here is a condensed summary of all debate rounds so far:\n{summary}\n\n"
        "Review the debate summary to understand what answers, reasoning, and disagreements have emerged. "
        "Use this summary to possibly update your answer. Re-analyze the problem carefully, incorporating any valid insights from the summary.\n"
        "The original math problem is:\n{problem}\n"
        "Explain your reasoning. Your final answer should be a single numerical number, "
        "in the form \\boxed{{answer}}, at the end of your response.\n"
        "At the very end, add a line: Confidence: XX%."
    ),
    "mmlu_full_history": (
        "Here is a condensed summary of all debate rounds so far:\n{summary}\n\n"
        "Review the debate summary to understand what answers and reasoning have been discussed. "
        "Use this summary to possibly update your answer. Re-evaluate all options, considering insights from the summary.\n"
        "Original question:\n{question}\n"
        "A) {A}\nB) {B}\nC) {C}\nD) {D}\n\n"
        "Explain your answer, putting the answer in the form (X) at the end of your response.\n"
        "At the very end, add a line: Confidence: XX%."
    ),
    "bio_full_history": (
        "Here is a condensed summary of all debate rounds so far:\n{summary}\n\n"
        "Review the debate summary to see what facts and contributions have been discussed. "
        "Use this summary to possibly update your biography. Re-examine your biography, incorporating accurate information from the summary.\n"
        "Provide an updated bullet point biography of {person} "
        "highlighting their contributions and achievements as a computer scientist, "
        "with each fact separated with a new line character.\n"
        "At the very end, add a line: Confidence: XX%."
    ),
    "chess_valid_full_history": (
        "Here is a condensed summary of all debate rounds so far:\n{summary}\n\n"
        "Review the debate summary to see what destination squares and justifications have been discussed. "
        "Use this summary to possibly update your answer. Re-check the chess position and valid moves, considering insights from the summary.\n"
        "Given the chess game:\n{moves}\n"
        "Give one valid destination square for the chess piece at {origin}.\n"
        "State the destination square in the form (X), where X follows the regex [a-h][1-8], for example (e5).\n"
        "Give a one line explanation of why your destination square is a valid move.\n"
        "At the very end, add a line: Confidence: XX%."
    ),
    "chess_move_full_history": (
        "Here is a condensed summary of all debate rounds so far:\n{summary}\n\n"
        "Review the debate summary to see what moves and strategic reasoning have been discussed. "
        "Use this summary to possibly update your answer. Using the chess suggestions from the summary as additional advice and your earlier generated solution, "
        "can you give me your updated thoughts on the best next chess move I should play given the chess sequence?\n"
        "Here is the current sequence of moves in a chess game:\n{moves}\n"
        "What is the best chess move I should execute next? Give a single move suggestion of the form 14. <XXX> "
        "and make sure the chess move is valid in the current board state.\n"
        "At the very end, add a line: Confidence: XX%."
    ),
    "bioasq_full_history": (
        "Here is a condensed summary of all debate rounds so far:\n{summary}\n\n"
        "Review the debate summary to see what answers and reasoning have been discussed. "
        "Use this summary to possibly update your answer. Re-evaluate the question and provide your final answer following the rules.\n"
        "Question: {question}\n"
        "Type: {qtype}  (one of factoid | yesno | list)\n\n"
        "Rules:\n"
        "- If type is 'factoid', return a short entity/string.\n"
        "- If type is 'yesno', return only 'yes' or 'no'.\n"
        "- If type is 'list', return a comma-separated list of items.\n\n"
        "On the LAST line, write: FINAL: <your answer>\n"
        "At the very end, add a line: Confidence: XX%."
    ),
}