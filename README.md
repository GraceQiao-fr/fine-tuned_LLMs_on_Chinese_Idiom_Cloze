# Fine-tuned LLMs on Chinese Idiom Cloze

## T5-small and mT5-small

### Methods 

We use T5-small because we want to see how the smaller model performs for the similar task. mT5-small was used to (roughly) see if the single Chinese language model can perform better than the multilingual language model.
After preprocessing the data and splitting the data into training and validation sets, we tokenized the input sentences and labels using the BertTokenizer to feed in the T5-small model, because it is pre-trained on Chinese to predict the missing single character. For more information, please refer to uer/t5-small-chinese-cluecorpussmall (https://huggingface.co/uer/t5-small-chinese-cluecorpussmall). This model has 6 layers and 512 hidden units.
We tokenized the input sentences and labels using the AutoTokenizer to feed in the T5-small model. For more information, please refer to google/mt5-small (https://huggingface.co/google/mt5-small). 
We used a custom collate function to pad the input and labels to the same length and to create attention masks for the input and labels. We fine-tuned these models on our Chinese idiom completion task by training it on a dataset of 20,000 examples for 5 epochs. During training, we used the Adam optimizer with a learning rate of 5e-5 and a batch size of 8. We used the cross-entropy loss function to compute the model's loss.
We evaluated our fine-tuned T5-small model on a validation set of 3,000 examples. We measured the model's performance mainly using accuracy. Our fine-tuned T5-small model achieved an accuracy of 40.70% on the validation set and 41.38% on the test set. Our fine-tuned mT5-small model achieved an accuracy of 43.32% on the validation set and 42.34% on the test set, which is better than T5-small pre-trained on Chinese only. The reason can be that the multilingual models perform better than unilingual models, but the evidence is not enough. It can also be caused by the reason of data quality, pre-training methods, etc.

### Experiments and Results 

**The Scale of Data:**

We tried to use 200,000 data to train the T5-small model, and it performs much better, with an accuracy of 70.04% on the validation set and 67.89% on the test set. It took around 5 hours to train the model. Therefore, we decided to control the training time and still use 20,000 as our training set scale. However, this shows that more data can make the effectiveness of fine tuning much better, at least on the T5-small model.
(*Note: the performance data are different from our presentation, because we retrain the models and the results changed a bit.)

**Batch Size:**

We tried batch size 4 for both models, it was too slow and did not perform better. We also tried batch size 16, and even if we trained for more epochs, the performance of both models were worse. So we finally used batch size 8 to train both models.

### Prompt Engineering:

**Pre-fine-tuning**

We tried to only use prompt engineering before fine-tuning, but both models got the results of 0% accuracy. It indicates that doing prompt engineering before fine-tuning makes no sense.

**Post-fine-tuning**

We also tried to use different instructions that have the similar meaning with the fine-tuning instruction to instruct the model to choose the proper idiom from the candidates, in order to see if the model can really understand the instruction. However, the accuracy became only 5.98% and 16.24% for T5-small and mT5-small, respectively. The output also showed that although the model can make some correct predictions, they do not really understand the instruction to choose the proper idiom. 

We can conclude that the proper prompts highly rely on the training data during fine-tuning. In other words, the models can only execute the same instructions as those in the training data. This shows that the instructions can be useless for both models, and we only need to provide the models with input and output results.
For more details of T5-small and mT5-small models, please refer to the notebook `T5/mT5-small: Prompt Engineering, Evaluation and Comparison` (https://github.ubc.ca/qmygrace/COLX_585_group_zootopia/blob/main/Final_result/T5%26mT5_small_propmt_engineering_and_comparison.ipynb) 

## ChatGPT Prompt Engineering

### Methods

We used the OpenAI API gpt-3.5-turbo to sent the request to ChatGPT and get responses.  The structure of our final prompts is:
* Two examples of choosing correct idiom(s) from the candidates given the paragraphs: One example included only 1 missing idiom from the paragraph, and another one include multiple missing idioms;
* The instruction which means “please follow the examples above and choose appropriate idiom(s) from the following brackets to fill in ‘#idiom#’”;
* The instruction which means “please only reply idioms, do not reply other characters”; 
* The candidates in brackets: Each group of the candidates of the same missing idiom is in the same bracket; and
* The paragraph with the missing idiom(s)
We tried to use 3,000 test data to evaluate. However, the fee was used up halfway. It was too time consuming and money consuming to restart and collect all the responses, so we decided to use what we had. We got 2,096 responses. The amount is not perfect, but okay to represent the performance on 3,000 test data.
We also did some minor cleaning on the response to clean up the mess signals.

### Experiment
We also tried once on only 60 requests, and in each prompt we only include one example, and did not enforce not to return other characters other than the idiom(s). The performance was okay of around 24%, but it tended to return the whole paragraph. Therefore, we revised the prompts.
The data amount 60 is also too small, and its accuracy may not be accurate, so we wanted to try the whole test set. As mentioned in the previous section “Method”, we eventually did not get 3,000 but 2,096 instead.

### Results

ChatGPT achieved an accuracy of 18.82%, which is quite low. After inspecting the mistakes, we found that some were due to it returning the whole paragraph instead of the idiom(s) only, but most were really wrongly chosen.
The reason may be the Chinese training data only consist of around 0.1% of the training data of GPT-3.5. Moreover,  Chinese idioms are quite difficult in Chinese, with certain structure and background stories and metaphor, making it more difficult to predict than general Chinese.
