from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata


class ChatPDF:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):
        self.model = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key="AIzaSyD8woTB9YBjkSr7qLhNCcfmRlmqCM6NIKo")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=100)
        self.prompt = PromptTemplate.from_template(
            """
          <s> [INST] You are an assistant for sprint performance analysis. Depending on the Phase and Gender Of the sprinter use the following Metrics  and pieces of context  to generate a comprehensive performance report.If you don't have the metrics, don't generate the report ,just say I am sorry, but I cannot generate a report for sprint performance analysis because the necessary metrics are not provided . Ensure the report follows this template exactly add imojis at the end of each section to make it more engaging.:
            **Stride Length**:
            - Metric: *Place_Holder* meters
            - Feedback: [Provide detailed feedback on whether it is above, below, or within the ideal range for the sprint category. Discuss how stride length affects overall speed and efficiency. Include recommendations for improving stride length if necessary, such as specific drills or exercises.]
            **Stride Frequency**:
            - Metric: *Place_Holder* strides/second
            - Feedback: [Provide detailed feedback on whether it is above, below, or within the ideal range. Explain the impact of stride frequency on sprinting performance. Offer suggestions for increasing stride frequency, including tips on technique and training regimens.]
            **Acceleration**:
            - Metric: *Place_Holder* m/s²
            - Feedback: [Provide detailed feedback on whether it is above, below, or within the ideal range. Discuss how acceleration influences the start and overall sprint performance. Recommend specific acceleration drills and strength training exercises to improve this metric.]
            **Velocity**:
            - Metric: *Place_Holder* m/s
            - Feedback: [Provide detailed feedback on whether it is above, below, or within the ideal range. Analyze how velocity impacts the athlete's performance in a sprint. Suggest strategies to increase velocity, including technical adjustments and endurance training.]
            **Touchdown**:
            - Metric: *Place_Holder* seconds
            - Feedback: [Provide detailed feedback on whether it is above, below, or within the ideal range. Explain how touchdown time affects stride efficiency and speed. Recommend ways to optimize touchdown time, such as improving ground contact time and leg strength.]
            **Overall Performance**:
            - Summary: [Provide an overall assessment of the performance. Highlight the athlete's strengths and identify areas that need improvement. Offer a balanced view, emphasizing what the athlete is doing well and where they can focus their efforts to achieve better results.]
            **Additional Insights**:
            - [Offer additional insights or observations that could help the athlete improve their sprint performance. This may include advice on technique adjustments, specific training suggestions, or considerations regarding equipment and gear. Discuss the importance of recovery and injury prevention, and provide tips on maintaining peak physical condition.]

            If you don't know the answer, just say that you don't know. Use a thorough and detailed approach, ensuring that each section is informative and provides actionable insights. [/INST] </s> 
            [INST] Metrics: 
           {question} 
            Context: {context} 
           
            Answer: [/INST]
            """
        )

    def ingest(self, pdf_file_path: str):
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        vector_store = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings())
        self.retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.0,
            },
        )

        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                      | self.prompt
                      | self.model
                      | StrOutputParser())

    def ask(self, query: str):
        if not self.chain:
            return "Please, add a PDF document first."

        return self.chain.invoke(query)

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None
