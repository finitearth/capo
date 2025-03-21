"""Task descriptions for each dataset."""

TASK_DESCRIPTIONS = {
    "sst-5": "The dataset consists of movie reviews with five levels of sentiment labels: very negative, negative, neutral, positive, and very positive. The task is to classify each movie review into one of these five sentiment categories. The class will be extracted between the markers <final_answer>answer</final_answer>.",
    "agnews": "The dataset contains news articles categorized into four classes: World, Sports, Business, and Sci/Tech. The task is to classify each news article into one of the four categories. The class will be extracted between the markers <final_answer>answer</final_answer>.",
    "subj": "The dataset contains sentences labeled as either subjective or objective. The task is to classify each sentence as either subjective or objective. The class will be extracted between the markers <final_answer>answer</final_answer>.",
    "gsm8k": "The dataset consists of grade school math word problems that require multi-step reasoning to solve. The task is to solve each word problem and provide the final answer. The final solution will be extracted between the markers <final_answer>answer</final_answer>.",
    "copa": "The dataset consists of premises and two possible choices for the effect or cause of the premise. The task is to determine which of the two choices (A or B) is the correct effect of the premise. The class will be extracted between the markers <final_answer>answer</final_answer>.",
}
