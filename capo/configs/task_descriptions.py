"""Task descriptions for each dataset."""

TASK_DESCRIPTIONS = {
    "sst-5": "The dataset consists of movie reviews with five levels of sentiment labels: veryNegative, negative, neutral, positive, and veryPositive. The task is to classify each movie review into one of these five sentiment categories. The class will be extracted between the markers <final_answer> answer </final_answer>.",
    "agnews": "The dataset contains news articles categorized into four classes: World, Sports, Business, and Sci/Tech. The task is to classify each news article into one of the four categories. The class will be extracted between the markers <final_answer> answer </final_answer>.",
    "subj": "The dataset contains sentences labeled as either subjective or objective. The task is to classify each sentence as either subjective or objective. The class will be extracted between the markers <final_answer> answer </final_answer>.",
    "rte": "The dataset contains pairs of sentences where the task is to determine whether the meaning of one sentence can be inferred from the other. The task is to classify each pair as either Entailment (if the second sentence follows logically from the first) or NoEntailment (if the second sentence does not necessarily follow from the first). The class will be extracted between the markers <final_answer> answer </final_answer>.",
    "gsm8k": "The dataset consists of grade school math word problems that require multi-step reasoning to solve. The task is to solve each word problem and provide the final answer. The final solution will be extracted between the markers <final_answer> answer </final_answer>.",
}
