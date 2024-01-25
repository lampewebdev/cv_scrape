import { ChatOllama } from "@langchain/community/chat_models/ollama";
import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { createRetrievalChain } from "langchain/chains/retrieval";
const loader = new CheerioWebBaseLoader(
    "https://docs.smith.langchain.com/overview"
);

const docs = await loader.load();
const splitter = new RecursiveCharacterTextSplitter();
const splitDocs = await splitter.splitDocuments(docs);

const chatModel = new ChatOllama({
    baseUrl: "http://localhost:11434", // Default value
    model: "mistral",
});

const embeddings = new OllamaEmbeddings({
    model: "mistral",
    maxConcurrency: 5,
});

const vectorstore = await MemoryVectorStore.fromDocuments(
    splitDocs,
    embeddings
);
const prompt =
    ChatPromptTemplate.fromTemplate(`Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}`);

const documentChain = await createStuffDocumentsChain({
    llm: chatModel,
    prompt,
});

const retriever = vectorstore.asRetriever();

const retrievalChain = await createRetrievalChain({
  combineDocsChain: documentChain,
  retriever,
});
const result = await retrievalChain.invoke({
  input: "what is LangSmith?",
});

console.log(result.answer);
// console.log("invoke: what is Stockholm?");
// const response = await chatModel.invoke("what is Stockholm?");
// console.log('response:', response.content)
