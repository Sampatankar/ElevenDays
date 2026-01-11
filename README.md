This repo is to contain the codebase that documents my learning in public for a 10/11 day curriculum related to a possible second-line interview with a successful MLOps consultancy.  The point of this sprint learn is to bring my production-grade Python, software engineering, ML pipelines, system design & data intensity and containerisation knowledge up to some capability.

The curriculum, developed in collaboration with Google Gemini, utilising books in my personal library, as at the end of December 2025 is as follows:

This 60-hour curriculum (6 hours/day over 10 days) is mapped to your specific books to prepare you for their technical standards.

Once I'm happy with my knowledge around my MLOps methods and pipelines, this project, especially phase 2, will be the knowledge-base upon which I can build to develop my ML Engineer/Research Scientist work.

---

## Phase 1: Production-Grade Python & Software Engineering (12 Hours)

**Objective:** Transition from "scripts" to "maintainable systems" that the company leads require.

* **Days 1 & 2 (12 Hours total):**
* **Book: *Fluent Python* (Ramalho)**
* **Read:** Part I (Prologue, Ch 1) on the "Python Data Model" and Part II (Ch 2-4) on "Data Structures".
* **Action:** Implement a custom dataset class using Pythonic "magic methods" (`__len__`, `__getitem__`) instead of manual indexing.


* **Book: *Effective Python* (Slatkin)**
* **Read:** Items 1-10 (Pythonic Thinking), Items 19-26 (Functions), and Items 75-84 (Testing and Debugging).
* **Action:** Rewrite an existing ML script to use **Type Hinting** and **List Comprehensions** for cleaner code.


* **Book: *Beyond the Basic Stuff with Python* (Sweigart)**
* **Read:** Part III (Practice Projects) focusing on "Type Hints" and "Effective Testing."
* **Action:** Set up a `pytest` suite for your data processing functions.


---

## Phase 2: Building Robust ML Pipelines (18 Hours)

**Objective:** Build reproducible pipelines, which the company identifies as a core bottleneck for their clients.

* **Days 3, 4, & 5 (18 Hours total):**
* **Book: *Hands-on Machine Learning* (Geron)**
* **Read:** Chapter 2 (End-to-End Machine Learning Project).
* **Action:** Follow the project but strictly implement everything inside **Scikit-Learn Pipelines** (`Pipeline` and `ColumnTransformer`) rather than separate cells.


* **Book: *Approaching (Almost) Any Machine Learning Problem* (Thakur)**
* **Read:** Chapters on "Cross Validation" and "Evaluation Metrics."
* **Action:** Implement advanced cross-validation strategies to ensure your model won't fail on unseen production data.


* **Book: *Software Engineering for Data Scientists* (Nelson)**
* **Read:** Sections on "Project Structure" and "Environment Management."
* **Action:** Organize your project into a modular directory structure (e.g., `data/`, `models/`, `src/`) similar to the Fuzzy Labs tech test repo.


---

## Phase 3: ML System Design & Data Intensity (18 Hours)

**Objective:** Understand the "plumbing" of ML systems—data versioning, tracking, and reliability.

* **Days 6, 7, & 8 (18 Hours total):**
* **Book: *Designing Machine Learning Systems* (Chip Huyen)**
* **Read:** Chapter 3 (Data Engineering) and Chapter 6 (Model Development and Offline Evaluation).
* **Action:** Design a schema for **Data Validation** (checking for nulls or distribution shifts) using principles from this book.


* **Book: *Designing Data-Intensive Applications* (Kleppmann)**
* **Read:** Chapter 1 (Reliable, Scalable, Maintainable Applications) and Chapter 10 (Batch Processing).
* **Action:** Sketch an architecture diagram for an ML system that separates training from inference, focusing on how data flows through the system.


* **Practical Lab (Open Source):**
* **Task:** Integrate **ZenML** and **MLflow** into your project.
* **Goal:** Create a 3-step ZenML pipeline: `ingest` → `train` → `evaluate`. Log the training parameters and the final model "artifact" to MLflow.


---

## Phase 4: Containerization & Final Integration (12 Hours)

**Objective:** Package your model for production, a "fluency" requirement for MLOps engineers.

* **Days 9 & 10 (12 Hours total):**
* **Book: *Software Engineering for Data Scientists* (Nelson)**
* **Read:** Sections on "Docker" and "Deployment."
* **Action:** Write a **Multi-stage Dockerfile** for your project. Stage 1 installs heavy dependencies (e.g., Scikit-learn); Stage 2 copies only the code and trained model to keep the final image small.


* **Final Synthesis:**
* **Action:** Wrap your trained model in a **FastAPI** or **Flask** server (similar to the Fuzzy Labs test server) and run it inside your Docker container.
* **Verification:** Ensure you can send a JSON request to your containerized server and receive a prediction back.


---

### Suggested Daily Breakdown (60 Hours Total)

| Day | Topic | Primary Book(s) | Focus Task |
| --- | --- | --- | --- |
| 1-2 | **Clean Python** | *Fluent Python*, *Effective Python* | Refactor scripts with types & tests. |
| 3-4 | **ML Pipelines** | *Hands-on ML*, *Approaching ML Problems* | Build Sklearn Pipelines for a regression task. |
| 5-6 | **System Design** | *Designing ML Systems*, *DDIA* | Create a data validation & schema plan. |
| 7-8 | **MLOps Tooling** | *SW Eng for Data Scientists*, ZenML Docs | Connect ZenML + MLflow to your pipeline. |
| 9 | **Docker** | *SW Eng for Data Scientists* | Write a production-grade Multi-stage Dockerfile. |
| 10 | **Final Polish** | All | Connect the API to the container; document the repo. |

While I may choose to follow where I can, I have to work in and around my internship with WeBuyBricks, so hours will be around that.
