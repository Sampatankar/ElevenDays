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


- Step 1: Load Data!
    - We will ingest data first but:
        - use design patterns to handle other sets of data accordingly
        - make it readable and reproducible in that sense
        - use 'factory' design patterns - where processing the data follows similar methods, only differentiating when needed
            - factory_design_pattern.py
            - ingest_data.py - implements the factory design pattern
    - a factory design pattern: 
        - this makes loading data 'smarter'
        - instead of your pipeline hard-coding how to read data (e.g. CSV vs JSON vs API vs Kafka), you ask the factory to give you the right ingestion object based on context
        - You ask for what you want, the factory decides how to do it
        - The factory pattern:
            - centralises source-specific logic
            - hides messy conditionals
            - makes it easy to add new data sources without touching the pipeline
            - enforces a consistent interface across ingestion types
            - the pipeline stays clean and complexity lives in one place
        - TLDR; so instead of a multitude of hard to test and easy to break if statements for each data object type, we define a common interface with a class and function, then implement concrete ingestors using more class and function methods
            - The factory itself is a small class and function set of if statements that call on the concrete ingestors which call on the common interface class
            - the ingestion call statement is then data type agnostic
                - https://chatgpt.com/share/6996fa89-87bc-800f-84d1-a18295214b29
                - factory_design_pattern.py in the explanations folder gives an example and explanation in code of the factory design pattern


- Strategy Pattern:
Imagine you're developing an e-commerce application.  Customers can choose different payment methods like CC, PayPal or Bitcoin.  Each payment method has a different implementation but the overall process is the same: the customer pays for the order.

    - PaymentMethod strategy is an interface that defines how payments are processed
    - CreditCardPayment, PayPalPayment, BitcoinPayment are concrete strategies that are different implementations of payment processing
    - ShoppingCart is a context that uses a payment method to process a customer's payment
    - example code in: explanations/strategy_design_pattern.py



    


