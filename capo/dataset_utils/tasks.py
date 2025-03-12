TASK_DESCRIPTIONS = {
    "sst-5": "The dataset consists of movie reviews with five levels of sentiment labels: veryNegative, negative, neutral, positive, and veryPositive. The task is to classify each movie review into one of these five sentiment categories. The class will be extracted between the markers <final_answer> answer </final_answer>.",
    "agnews": "The dataset contains news articles categorized into four classes: World, Sports, Business, and Sci/Tech. The task is to classify each news article into one of the four categories. The class will be extracted between the markers <final_answer> answer </final_answer>.",
    "subj": "The dataset contains sentences labeled as either subjective or objective. The task is to classify each sentence as either subjective or objective. The class will be extracted between the markers <final_answer> answer </final_answer>.",
    "rte": "The dataset contains pairs of sentences where the task is to determine whether the meaning of one sentence can be inferred from the other. The task is to classify each pair as either Entailment (if the second sentence follows logically from the first) or NoEntailment (if the second sentence does not necessarily follow from the first). The class will be extracted between the markers <final_answer> answer </final_answer>.",
    "gsm8k": "The dataset consists of grade school math word problems that require multi-step reasoning to solve. The task is to solve each word problem and provide the final answer. The final solution will be extracted between the markers <final_answer> answer </final_answer>.",
}

# Initial prompts for each task where created using the Claude Sonnet 3.7 API https://claude.ai/,
# prompting it with the following instructions and the task descriptions above:
# """
# Please create diverse prompts for the following task. They should be linguistically diverse
# (but always in English) and have varying lengths and complexities. This means some consist
# only of a short sentence with a rather high-level description while others elaborate on the
# task in little more detail.
#
# Task: <task_description>
#
# Explicitly state this expected format as part of the prompts. Create overall 20 prompts
# within quotes as an array:
# """
#
# The corresponding chats with Claude Sonnet 3.7 API are given below:
# sst-5: https://claude.ai/share/b3087202-4aea-4f16-9464-0ed2056c7ec6
# agnews: https://claude.ai/share/7d8d866f-00eb-480d-b117-355f48b818e1
# subj: https://claude.ai/share/c11b3edb-e5a0-4c1f-942b-42acacd0bc2b
# rte: https://claude.ai/share/29166259-0ef0-4cd5-8a84-c68496a1ba6e
# gsm8k: https://claude.ai/share/94483a7d-9388-4671-bb73-086ef198fce3


INITIAL_PROMPTS = {
    "sst-5": [
        "Classify this movie review's sentiment as veryNegative, negative, neutral, positive, or veryPositive. Put your answer between <final_answer> </final_answer> tags.",
        "What's the sentiment of this film review? Choose from: veryNegative, negative, neutral, positive, or veryPositive. Format your response with <final_answer> </final_answer>.",
        "Determine the emotional tone of the following movie critique. Is it veryNegative, negative, neutral, positive, or veryPositive? Your classification must be provided between <final_answer> and </final_answer> markers.",
        "Sentiment analysis task: categorize this cinema review as veryNegative, negative, neutral, positive, or veryPositive. Include your final classification within <final_answer> </final_answer>.",
        "Read the movie review and identify its sentiment. Select from these five categories: veryNegative, negative, neutral, positive, or veryPositive. Place your answer inside <final_answer> </final_answer>.",
        "Analyze the sentiment expressed in this film critique. Categorize it as either veryNegative, negative, neutral, positive, or veryPositive, and present your answer between <final_answer> </final_answer> tags.",
        "Quick sentiment check - is this movie review veryNegative, negative, neutral, positive, or veryPositive? Answer within <final_answer> </final_answer>.",
        "Evaluate the emotional content of the following film review and classify it into one of five sentiment categories: veryNegative, negative, neutral, positive, or veryPositive. Your classification must be provided between <final_answer> and </final_answer> markers.",
        "Given this movie critique, determine whether the overall sentiment is veryNegative, negative, neutral, positive, or veryPositive. Express your answer using the required format: <final_answer> chosen_category </final_answer>.",
        "What sentiment does this movie review convey? Pick from veryNegative, negative, neutral, positive, or veryPositive. Remember to format as <final_answer> your_classification </final_answer>.",
        "Assess the tone of the provided film review and categorize it as one of the following: veryNegative, negative, neutral, positive, or veryPositive. Your final classification must appear between <final_answer> and </final_answer> tags.",
        "Based on careful reading of this movie review, assign it to one of these sentiment categories: veryNegative, negative, neutral, positive, or veryPositive. Present your answer within <final_answer> </final_answer>.",
        "Classify the sentiment in this cinema critique using a five-point scale: veryNegative, negative, neutral, positive, or veryPositive. Your answer must be enclosed within <final_answer> </final_answer>.",
        "How would you describe the sentiment of this movie review? Choose from veryNegative, negative, neutral, positive, or veryPositive. Place your chosen category between <final_answer> and </final_answer>.",
        "I need you to determine whether the sentiment of this film review is veryNegative, negative, neutral, positive, or veryPositive. Your final answer should be formatted like this: <final_answer> sentiment_category </final_answer>.",
        "Please analyze this movie review and place it in one of five sentiment categories (veryNegative, negative, neutral, positive, or veryPositive). Format your response with <final_answer> followed by your categorization, then </final_answer>.",
        "Movie review sentiment classification task: From the following five options - veryNegative, negative, neutral, positive, or veryPositive - which best describes this review? Your answer must appear between <final_answer> and </final_answer> markers.",
        "Examine the following film critique and determine its emotional valence. Options are veryNegative, negative, neutral, positive, or veryPositive. Ensure your answer follows this format: <final_answer> selected_sentiment </final_answer>.",
        "Review the text and decide which sentiment category applies: veryNegative, negative, neutral, positive, or veryPositive. Your classification must be provided between <final_answer> </final_answer> tags.",
        "Sentiment detection: Read this movie review carefully and identify whether it expresses a veryNegative, negative, neutral, positive, or veryPositive sentiment. Your final classification should be presented as <final_answer> classification </final_answer>.",
    ],
    "agnews": [
        "Classify this news article into one of these categories: World, Sports, Business, or Sci/Tech. Put your answer between <final_answer> tags.",
        "Read the following news article and determine if it belongs to World, Sports, Business, or Sci/Tech. Your classification should be placed within <final_answer> tags.",
        "Which category does this news article belong to? Choose from World, Sports, Business, or Sci/Tech and provide your answer between <final_answer> </final_answer> markers.",
        "I need you to classify this news content into one of four categories (World, Sports, Business, Sci/Tech). Place only your final classification within <final_answer> </final_answer> tags.",
        "Determine the appropriate category for the following news article. Options are World, Sports, Business, and Sci/Tech. Format your response with <final_answer> category </final_answer>.",
        "Please read this news article carefully and assign it to one of these four categories: World, Sports, Business, or Sci/Tech. Your answer must be formatted as <final_answer> category </final_answer>.",
        "Based on the content of this news article, classify it as either World, Sports, Business, or Sci/Tech. Your classification must be placed between <final_answer> </final_answer> tags for proper extraction.",
        "Analyze this news article and identify whether it belongs to World, Sports, Business, or Sci/Tech categories. Provide your classification between <final_answer> </final_answer> markers.",
        "News article classification task: Categorize the following text as World, Sports, Business, or Sci/Tech. Your answer should be formatted as <final_answer> category </final_answer>.",
        "You are a news categorization system. Read the article below and assign it to one of these categories: World, Sports, Business, or Sci/Tech. Format: <final_answer> category </final_answer>",
        "As an AI assistant, please help classify this news article into one of the following four categories: World, Sports, Business, or Sci/Tech. Remember to place your classification within <final_answer> </final_answer> tags.",
        "Read the following news text and determine which category it belongs to. Choose from: World, Sports, Business, or Sci/Tech. Your final answer must be enclosed in <final_answer> </final_answer> tags for automated extraction.",
        "Given this news article, what category does it fall under? Select from World, Sports, Business, or Sci/Tech. Ensure your answer is formatted as <final_answer> category </final_answer>.",
        "I'm working on a news classification project. Could you read this article and tell me if it's about World, Sports, Business, or Sci/Tech? Please put your answer between <final_answer> </final_answer> markers.",
        "Classification task: Analyze the news content below and determine its category (World, Sports, Business, or Sci/Tech). For proper data extraction, format your answer as <final_answer> category </final_answer>.",
        "Your task is to categorize the following news article into exactly one of these four classes: World, Sports, Business, or Sci/Tech. The classification must be provided between <final_answer> </final_answer> tags to be properly processed by our system.",
        "Please review this news article carefully. Based on its content, assign it to the most appropriate category among World, Sports, Business, and Sci/Tech. To ensure your answer is correctly processed, place it within <final_answer> </final_answer> tags.",
        "Examine the following news text and identify whether it should be classified as World, Sports, Business, or Sci/Tech content. Your classification must be formatted with <final_answer> tags for automated extraction by our system.",
        "We're building a news classifier and need your help categorizing articles. Read the text below and decide if it belongs to World, Sports, Business, or Sci/Tech. Important: place your single-word answer inside <final_answer> </final_answer> tags.",
        "In our dataset of news articles, each piece must be classified into one of four categories: World, Sports, Business, or Sci/Tech. After reading the article below, determine its appropriate category and ensure you format your answer as <final_answer> category </final_answer> for our extraction script.",
    ],
    "subj": [
        "Determine if this sentence is subjective or objective and put your answer between <final_answer> tags.",
        "Classify the given sentence as either subjective (expressing personal opinions, emotions, or judgments) or objective (stating factual information without personal bias). Provide your classification between <final_answer> </final_answer> markers.",
        "Is the following text subjective or objective? Answer with just the word 'subjective' or 'objective' inside <final_answer> </final_answer> tags.",
        "Read this sentence and decide: is it expressing facts (objective) or opinions (subjective)? Your classification should be placed between <final_answer> </final_answer>.",
        "Subjectivity analysis task: Examine the sentence and determine if it conveys factual information (objective) or personal opinions/feelings (subjective). Format your answer as <final_answer>objective</final_answer> or <final_answer>subjective</final_answer>.",
        "Quick classification needed: subjective or objective? Place your one-word answer within <final_answer> </final_answer>.",
        "Analyze whether the given text presents factual information or expresses personal views. Respond with either 'subjective' or 'objective' between the <final_answer> </final_answer> markers.",
        "Your task is to evaluate the sentence and decide if it contains objective information (facts, measurable data) or subjective content (opinions, judgments, emotions). Return only 'subjective' or 'objective' inside <final_answer> tags.",
        "Sentence classification task: Does the sentence state facts (objective) or express opinions/feelings (subjective)? Provide your answer using the format <final_answer>your_answer</final_answer>.",
        "Determine the nature of this sentence - is it presenting factual, verifiable information (objective) or personal viewpoints, feelings, or judgments (subjective)? Respond with only 'objective' or 'subjective' between <final_answer> </final_answer> tags.",
        "Review the text and classify: subjective or objective? Format: <final_answer>classification</final_answer>",
        "Carefully examine this sentence to determine whether it expresses an objective statement (factual, unbiased information that could be verified) or a subjective statement (personal opinions, judgments, or emotions that may vary from person to person). Provide your assessment between <final_answer> </final_answer> markers.",
        "Classify as objective (fact-based) or subjective (opinion-based). Answer within <final_answer> </final_answer>.",
        "Using your understanding of subjectivity vs. objectivity in language, determine if the given sentence is objective (states facts, provides information without personal bias) or subjective (expresses opinions, emotions, or personal judgments). Place your classification between the <final_answer> </final_answer> tags.",
        "Is this sentence stating facts (objective) or expressing opinions (subjective)? Answer using the required format: <final_answer>your_answer</final_answer>",
        "Linguistic analysis: Examine the provided sentence and determine whether it's objective (factual, unbiased, could be verified) or subjective (opinion-based, contains judgments, emotions, or personal perspective). Your response should be formatted as <final_answer>objective</final_answer> or <final_answer>subjective</final_answer>.",
        "Read the following sentence and classify it as either 'subjective' (containing opinions, judgments, or emotions) or 'objective' (presenting verifiable facts without personal bias). Your answer must be formatted as: <final_answer>your classification</final_answer>",
        "Subjective vs. objective classification: Analyze whether the sentence presents factual information or personal opinions/feelings. Place your one-word answer (either 'subjective' or 'objective') between <final_answer> and </final_answer> tags.",
        "Evaluate this sentence and determine if it's presenting objective information (facts that can be verified) or subjective content (opinions, judgments, or emotions). Provide your classification inside <final_answer> </final_answer> markers.",
        "Text classification task: decide if the sentence is objective (fact-based, unbiased, verifiable) or subjective (opinion-based, contains personal judgments or feelings). Format your answer as <final_answer>objective</final_answer> or <final_answer>subjective</final_answer>.",
    ],
    "rte": [
        "Determine if the second sentence logically follows from the first. Answer with 'Entailment' or 'NoEntailment' between <final_answer> tags.",
        "Analyze these two sentences and decide if one entails the other. Put your answer (Entailment/NoEntailment) inside <final_answer> </final_answer> markers.",
        "Does sentence B logically follow from sentence A? Classify as Entailment or NoEntailment and place your answer between <final_answer> tags.",
        "For the given pair of sentences, determine if the meaning of the second sentence can be inferred from the first. Provide your classification (Entailment/NoEntailment) within <final_answer> </final_answer> tags.",
        "Textual entailment task: Review the sentence pair and decide whether the second sentence necessarily follows from the first. Your answer must be either 'Entailment' or 'NoEntailment' enclosed in <final_answer> </final_answer>.",
        "Simple classification task: Does sentence 2 logically follow from sentence 1? Answer with Entailment or NoEntailment between <final_answer> tags.",
        "Evaluate whether the information in the second sentence can be inferred from the first sentence. Classify as either Entailment or NoEntailment and provide your answer within <final_answer> </final_answer> markers.",
        "Given two sentences, determine if the second one can be logically inferred from the first. Your classification (Entailment/NoEntailment) should be placed inside <final_answer> </final_answer>.",
        "Linguistic inference challenge: Can the meaning of sentence B be derived from sentence A? Indicate Entailment or NoEntailment in your response, surrounded by <final_answer> </final_answer> tags.",
        "Read these two sentences carefully and determine if there's a logical entailment relationship between them. Respond with either 'Entailment' or 'NoEntailment' inside <final_answer> </final_answer> markers.",
        "Natural language inference task: Decide if the second statement is entailed by the first. Your answer should be either Entailment or NoEntailment, placed between <final_answer> tags.",
        "Looking at this sentence pair, would a reasonable person infer the second from the first? Label as Entailment or NoEntailment and include your answer within <final_answer> </final_answer>.",
        "Sentence relationship analysis: Does the information in sentence 1 necessarily imply the information in sentence 2? Classify as Entailment or NoEntailment between <final_answer> </final_answer> tags.",
        "Determine the logical relationship between these sentences. If the second sentence logically follows from the first, classify as 'Entailment'; otherwise, classify as 'NoEntailment'. Place your answer inside <final_answer> </final_answer> markers.",
        "Evaluate this pair of sentences. If the truth of the first sentence guarantees the truth of the second, answer 'Entailment'; otherwise, answer 'NoEntailment'. Your classification must appear between <final_answer> tags.",
        "Analyze the semantic relationship between these two sentences. Does the meaning of sentence A entail the meaning of sentence B? Respond with Entailment or NoEntailment inside <final_answer> </final_answer>.",
        "Consider these two statements. If statement 2 is necessarily true when statement 1 is true, classify as Entailment; otherwise, classify as NoEntailment. Format your answer with <final_answer> </final_answer> tags.",
        "Logical inference detection: Examine whether the second sentence is a logical consequence of the first. Your classification (Entailment/NoEntailment) should be placed between <final_answer> </final_answer> markers.",
        "For this pair of sentences, your task is to determine whether the information in the second sentence can be inferred from the information provided in the first sentence. If yes, the classification is 'Entailment'; if no, the classification is 'NoEntailment'. Please provide your answer between <final_answer> </final_answer> tags.",
        "Assessment of textual inference: Does sentence 2 follow logically from sentence 1? If it does, the answer is Entailment; if not, the answer is NoEntailment. Place your classification inside <final_answer> </final_answer> markers.",
    ],
    "gsm8k": [
        "Solve this math word problem using step-by-step reasoning. Put your final answer between <final_answer> tags."
        "I need help with a multi-step grade school math problem. Can you walk through the solution and place the final result inside <final_answer> answer </final_answer> markers?"
        "Break down this word problem carefully. Remember to format the final answer within <final_answer> </final_answer> tags."
        "Analyze and solve the following elementary school math problem that requires multiple steps of reasoning. Your final answer must be contained within <final_answer> </final_answer> tags."
        "Please solve this math story problem. Show each step of your thought process. Be sure to include your final answer between <final_answer> and </final_answer> markers."
        "Work through this grade school word problem methodically. After your explanation, provide the ultimate answer surrounded by <final_answer> tags."
        "I'm struggling with this math word problem that needs multiple steps to solve. Can you help? Make sure to put your final answer between <final_answer> </final_answer> tags so I can easily find it."
        "Tackle this elementary math problem by breaking it into logical steps. When you reach the solution, enclose the final answer with <final_answer> and </final_answer> markers for clarity."
        "Solve step-by-step. Final answer between <final_answer> tags."
        "This multi-step math problem requires careful reasoning. Please work through it systematically and clearly mark your final answer using <final_answer> </final_answer> format."
        "Calculate the solution to this grade school word problem. Show your reasoning process, and don't forget to place your final answer within <final_answer> </final_answer> markers as required."
        "Can you help solve this math word problem? It may take several steps. When you've figured it out, please put the final answer in <final_answer> </final_answer> tags."
        "I'd appreciate your assistance with this elementary math word problem. Please explain each step of your reasoning and make sure to format the final solution inside <final_answer> </final_answer> markers."
        "Solve this math problem by breaking it down into smaller parts. Once you reach the conclusion, ensure your final answer appears between <final_answer> and </final_answer> tags."
        "Work through this word problem step-by-step. I need the answer formatted as <final_answer>your answer</final_answer> at the end of your explanation."
        "Calling all math wizards! I need help with this tricky multi-step word problem. Walk me through your solution process and wrap the final answer with <final_answer> </final_answer> tags."
        "Please analyze this elementary school math problem that requires multiple logical steps. After explaining your reasoning, provide the ultimate solution between <final_answer> tags."
        "I'm having trouble with this grade school math word problem. Can you provide a detailed solution? Make sure to put your final answer between <final_answer> and </final_answer> markers so I can easily identify it."
        "This problem requires multi-step reasoning to solve correctly. Please walk through your approach and clearly indicate your final answer using the <final_answer> </final_answer> format."
        "Solve the following math word problem by working through it methodically. Your explanation should be clear, and your final answer must be enclosed within <final_answer> </final_answer> tags as specified."
    ],
}
