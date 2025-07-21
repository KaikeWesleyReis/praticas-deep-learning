# Challenge

The attached python notebook implements a simple ecommerce search engine using Tf-IDF.

It uses the WANDS dataset from Wayfair to evaluate the performance of the search engine using **Mean Average Precision@10 (MAP)**.

**Important** - The assignment doesn’t require you to choose the best embedding model, rather choose and approach and discuss pros and cons and what you would like to do if you have more time and resources. **It’s more about your approach than the final answer.**

Test Duration: 90 - 120 minutes.

Please provide responses to the following prompts:

1) The search engine in the notebook has a MAP@10 across all queries of 0.29. This is considered low. Please propose some updates to increase the score. For reference, large ecommerce websites have MAP@10 values between 0.6—0.8, although there is no expectation for your solution to be in that range. The strength of your ideas holds greater weight than the final MAP score of the solution.
2) Currently, partial matches are treated as irrelevants, which penalizes the model too strictly. Can you implement another function that leverages the partial match count to provide a fairer assessment of performance? Please provide a justification for why you chose this function and the tradeoffs. If you choose to implement additional evaluation metrics, please provide a justification for using them along with tradeoffs.
3) For this prompt you can choose one of two options, but you DO NOT need to do both. We value the ability to improve model performance and refactor code equally, so please choose based on what you feel most comfortable doing:

        A) Please implement at least one change you suggest for prompt 1 to demonstrate an improvement in the MAP score. Please document your code changes with comments and markdown cells so we can follow your thought process.

    or

        B) Please modify the code to make it more object oriented, more flexible to accommodate changes to the retrieval model, and getting it ready for production (such as adding logging, error handling, etc.)

Please ensure your version of the notebook is reproducible in its entirety. We recommend restarting the kernel and running all the cells to ensure your results are reproducible before emailing the file back to us.