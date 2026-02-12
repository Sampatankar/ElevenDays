Project derived from this YouTube video: https://www.youtube.com/watch?v=o6vbe5G7xNo&t=98s

An End-to-end ML Project which uses ZenML and MLFlow (open source MLOps) from FreeCodeCamp

- basic houseprice prediction system

- What do most people do?
    - Limited exploration:
        - Most practitioners start with basic exploratory data analysis (EDA) using standard frameworks
        - They quickly move on to calling .fit on a model w/o thoroughly understanding the data
    - Basic model training:
        - After EDA they typically split the data, train a model and call .predict
        - The project often ends here with a focus on achieving high accuracy or minimising error
    - Lack of iteration:
        - Once the model is trained, it's rarely revisited or improved based on deeper insights from the data
        - There's minimal to no effort in validating assumptions or handling model violations


- This project's Approach:
    - Thorough Data Research:
        - Most practitioners start with basic EDA using standard frameworks (as above)
        - They quickly move in to calling .fit on a model w/o thoroughly understanding the data (as above)
    - Structured Data Processing:
        - Implement findings from EDA in the preprocessing stage, ensuring the data is clean and feature-engineered to maximise model performance
        - Continuously validate and correct assumptions during model training, fixing any violations through iterative improvment
    - Beyond Core ML:
        - We don't just train a model, we ensure it meets all necessary assumptions and refine it iteratively
        - We focus on building a robust pipeline that can be easily reproduced and deployed
    - MLOps & Production Readiness:
        - Differentiate our project by integrating MLOps practices using ZenML and MLFlow
        - Implement CI/CD pipelines to automate testing, deployment of the model in production
        - Ensure the model is not only accurate but also maintainable, scalable and ready for real-world use





