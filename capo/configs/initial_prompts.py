"""
Initial prompts for each dataset.

The initial prompts for each dataset where created using the Claude Sonnet 3.7 API https://claude.ai/,
prompting it with the following instructions and the task descriptions above:

'''
Please create diverse prompts for the following task. They should be linguistically diverse
(but always in English) and have varying lengths and complexities. This means some consist
only of a short sentence with a rather high-level description while others elaborate on the
task in little more detail.

Task: <task_description>

Explicitly state this expected format as part of the prompts. Create overall 20 prompts
within quotes as an array:
'''

Small changes were adapted after the initial prompts were generated to ensure that the prompts
are using the correct target labels.
"""


INITIAL_PROMPTS = {
    "sst-5": [
        "What's the sentiment of this film review? Choose from: very negative, negative, neutral, positive, or very positive. Format your response with <final_answer> </final_answer>.",
        "Determine the emotional tone of the following movie critique. Is it very negative, negative, neutral, positive, or very positive? Your classification must be provided between <final_answer> and </final_answer> markers.",
        "Sentiment analysis task: categorize this cinema review as very negative, negative, neutral, positive, or very positive. Include your final classification within <final_answer> </final_answer>.",
        "Read the movie review and identify its sentiment. Select from these five categories: very negative, negative, neutral, positive, or very positive. Place your answer inside <final_answer> </final_answer>.",
        "Analyze the sentiment expressed in this film critique. Categorize it as either very negative, negative, neutral, positive, or very positive, and present your answer between <final_answer> </final_answer> tags.",
        "Quick sentiment check - is this movie review very negative, negative, neutral, positive, or very positive? Answer within <final_answer> </final_answer>.",
        "Evaluate the emotional content of the following film review and classify it into one of five sentiment categories: very negative, negative, neutral, positive, or very positive. Your classification must be provided between <final_answer> and </final_answer> markers.",
        "Given this movie critique, determine whether the overall sentiment is very negative, negative, neutral, positive, or very positive. Express your answer using the required format: <final_answer> chosen_category </final_answer>.",
        "What sentiment does this movie review convey? Pick from very negative, negative, neutral, positive, or very positive. Remember to format as <final_answer> your_classification </final_answer>.",
        "Assess the tone of the provided film review and categorize it as one of the following: very negative, negative, neutral, positive, or very positive. Your final classification must appear between <final_answer> and </final_answer> tags.",
        "Classify the sentiment in this cinema critique using a five-point scale: very negative, negative, neutral, positive, or very positive. Your answer must be enclosed within <final_answer> </final_answer>.",
        "I need you to determine whether the sentiment of this film review is very negative, negative, neutral, positive, or very positive. Your final answer should be formatted like this: <final_answer> sentiment_category </final_answer>.",
        "Movie review sentiment classification task: From the following five options - very negative, negative, neutral, positive, or very positive - which best describes this review? Your answer must appear between <final_answer> and </final_answer> markers.",
        "Review the text and decide which sentiment category applies: very negative, negative, neutral, positive, or very positive. Your classification must be provided between <final_answer> </final_answer> tags.",
        "Sentiment detection: Read this movie review carefully and identify whether it expresses a very negative, negative, neutral, positive, or very positive sentiment. Your final classification should be presented as <final_answer> classification </final_answer>.",
    ],
    "agnews": [
        "Classify this news article into one of these categories: World, Sports, Business, or Sci/Tech. Put your answer between <final_answer> tags.",
        "Read the following news article and determine if it belongs to World, Sports, Business, or Sci/Tech. Your classification should be placed within <final_answer> tags.",
        "I need you to classify this news content into one of four categories (World, Sports, Business, Sci/Tech). Place only your final classification within <final_answer> </final_answer> tags.",
        "Please read this news article carefully and assign it to one of these four categories: World, Sports, Business, or Sci/Tech. Your answer must be formatted as <final_answer> category </final_answer>.",
        "Based on the content of this news article, classify it as either World, Sports, Business, or Sci/Tech. Your classification must be placed between <final_answer> </final_answer> tags for proper extraction.",
        "News article classification task: Categorize the following text as World, Sports, Business, or Sci/Tech. Your answer should be formatted as <final_answer> category </final_answer>.",
        "You are a news categorization system. Read the article below and assign it to one of these categories: World, Sports, Business, or Sci/Tech. Format: <final_answer> category </final_answer>",
        "As an AI assistant, please help classify this news article into one of the following four categories: World, Sports, Business, or Sci/Tech. Remember to place your classification within <final_answer> </final_answer> tags.",
        "Read the following news text and determine which category it belongs to. Choose from: World, Sports, Business, or Sci/Tech. Your final answer must be enclosed in <final_answer> </final_answer> tags for automated extraction.",
        "Given this news article, what category does it fall under? Select from World, Sports, Business, or Sci/Tech. Ensure your answer is formatted as <final_answer> category </final_answer>.",
        "I'm working on a news classification project. Could you read this article and tell me if it's about World, Sports, Business, or Sci/Tech? Please put your answer between <final_answer> </final_answer> markers.",
        "Classification task: Analyze the news content below and determine its category (World, Sports, Business, or Sci/Tech). For proper data extraction, format your answer as <final_answer> category </final_answer>.",
        "Your task is to categorize the following news article into exactly one of these four classes: World, Sports, Business, or Sci/Tech. The classification must be provided between <final_answer> </final_answer> tags to be properly processed by our system.",
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
        "Your task is to evaluate the sentence and decide if it contains objective information (facts, measurable data) or subjective content (opinions, judgments, emotions). Return only 'subjective' or 'objective' inside <final_answer> tags.",
        "Sentence classification task: Does the sentence state facts (objective) or express opinions/feelings (subjective)? Provide your answer using the format <final_answer>your_answer</final_answer>.",
        "Determine the nature of this sentence - is it presenting factual, verifiable information (objective) or personal viewpoints, feelings, or judgments (subjective)? Respond with only 'objective' or 'subjective' between <final_answer> </final_answer> tags.",
        "Carefully examine this sentence to determine whether it expresses an objective statement (factual, unbiased information that could be verified) or a subjective statement (personal opinions, judgments, or emotions that may vary from person to person). Provide your assessment between <final_answer> </final_answer> markers.",
        "Classify as objective (fact-based) or subjective (opinion-based). Answer within <final_answer> </final_answer>.",
        "Using your understanding of subjectivity vs. objectivity in language, determine if the given sentence is objective (states facts, provides information without personal bias) or subjective (expresses opinions, emotions, or personal judgments). Place your classification between the <final_answer> </final_answer> tags.",
        "Is this sentence stating facts (objective) or expressing opinions (subjective)? Answer using the required format: <final_answer>your_answer</final_answer>",
        "Read the following sentence and classify it as either 'subjective' (containing opinions, judgments, or emotions) or 'objective' (presenting verifiable facts without personal bias). Your answer must be formatted as: <final_answer>your classification</final_answer>",
        "Evaluate this sentence and determine if it's presenting objective information (facts that can be verified) or subjective content (opinions, judgments, or emotions). Provide your classification inside <final_answer> </final_answer> markers.",
        "Text classification task: decide if the sentence is objective (fact-based, unbiased, verifiable) or subjective (opinion-based, contains personal judgments or feelings). Format your answer as <final_answer>objective</final_answer> or <final_answer>subjective</final_answer>.",
    ],
    "gsm8k": [
        "Solve this math word problem using step-by-step reasoning. Put your final answer between <final_answer> tags.",
        "I need help with a multi-step grade school math problem. Can you walk through the solution and place the final result inside <final_answer> answer </final_answer> markers?",
        "Break down this word problem carefully. Remember to format the final answer within <final_answer> </final_answer> tags.",
        "Analyze and solve the following elementary school math problem that requires multiple steps of reasoning. Your final answer must be contained within <final_answer> </final_answer> tags.",
        "Please solve this math story problem. Show each step of your thought process. Be sure to include your final answer between <final_answer> and </final_answer> markers.",
        "I'm struggling with this math word problem that needs multiple steps to solve. Can you help? Make sure to put your final answer between <final_answer> </final_answer> tags so I can easily find it.",
        "Tackle this elementary math problem by breaking it into logical steps. When you reach the solution, enclose the final answer with <final_answer> and </final_answer> markers for clarity.",
        "Solve step-by-step. Final answer between <final_answer> tags.",
        "I'd appreciate your assistance with this elementary math word problem. Please explain each step of your reasoning and make sure to format the final solution inside <final_answer> </final_answer> markers.",
        "Work through this word problem step-by-step. I need the answer formatted as <final_answer>your answer</final_answer> at the end of your explanation.",
        "Calling all math wizards! I need help with this tricky multi-step word problem. Walk me through your solution process and wrap the final answer with <final_answer> </final_answer> tags.",
        "Please analyze this elementary school math problem that requires multiple logical steps. After explaining your reasoning, provide the ultimate solution between <final_answer> tags.",
        "I'm having trouble with this grade school math word problem. Can you provide a detailed solution? Make sure to put your final answer between <final_answer> and </final_answer> markers so I can easily identify it.",
        "This problem requires multi-step reasoning to solve correctly. Please walk through your approach and clearly indicate your final answer using the <final_answer> </final_answer> format.",
        "Solve the following math word problem by working through it methodically. Your explanation should be clear, and your final answer must be enclosed within <final_answer> </final_answer> tags as specified.",
    ],
    "copa": [
        "Choose the most logical cause or effect between options A and B. Provide your answer as either <final_answer>A</final_answer> or <final_answer>B</final_answer>.",
        "Determine which option follows logically from the given premise. Is it A or B? Your answer must be formatted as <final_answer>A</final_answer> or <final_answer>B</final_answer>.",
        "Select the statement that represents the most reasonable causal relationship to the given context. Respond with <final_answer>A</final_answer> or <final_answer>B</final_answer> only.",
        "Using commonsense reasoning, identify whether option A or option B is the correct cause or effect for the given scenario. Format your answer as <final_answer>A</final_answer> or <final_answer>B</final_answer>.",
        "Based on causal reasoning, which is more plausible: A or B? Enclose your answer with <final_answer> tags like this: <final_answer>A</final_answer> or <final_answer>B</final_answer>.",
        "Evaluate the two alternatives and select the one that represents the most logical causal relationship. Your response should be structured as <final_answer>A</final_answer> or <final_answer>B</final_answer>.",
        "Which option makes more sense as a cause or effect? Answer with <final_answer>A</final_answer> or <final_answer>B</final_answer>.",
        "From a commonsense perspective, analyze the given scenario and determine if A or B is the more reasonable cause/effect. Please provide your answer in the format <final_answer>A</final_answer> or <final_answer>B</final_answer>.",
        "Read the premise carefully. Then decide whether A or B is the more logical cause or effect. Your answer must be formatted as: <final_answer>A</final_answer> or <final_answer>B</final_answer>.",
        "For the Bala-COPA task, you need to utilize commonsense knowledge to determine which option (A or B) is causally related to the given statement. After your reasoning, provide only <final_answer>A</final_answer> or <final_answer>B</final_answer>.",
        "Pick A or B based on which has the stronger causal connection to the provided context. Ensure your answer is formatted as <final_answer>A</final_answer> or <final_answer>B</final_answer>.",
        "This is a causal reasoning task. Consider the premise, then select which option (A or B) is the most logical cause or effect. Your final answer must be enclosed in tags like this: <final_answer>A</final_answer> or <final_answer>B</final_answer>.",
        "Given the premise, determine the most plausible causal relationship. Is it option A or option B? Please format your answer as <final_answer>A</final_answer> or <final_answer>B</final_answer>.",
        "The Bala-COPA dataset tests commonsense causal reasoning abilities. Review the given scenario and decide whether Text A or Text B is the correct cause/effect. Your answer must be either <final_answer>A</final_answer> or <final_answer>B</final_answer>.",
        "Assess the causal relationship in the given context. Choose between options A and B, and provide your selection in the format <final_answer>A</final_answer> or <final_answer>B</final_answer>.",
    ],
}
