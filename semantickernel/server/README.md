# Phi-3-Mini for Semantic Kernel .NET Developer


*Support Semantic-Kernel 1.7.1*

At this stage, it is adapted for macOS and Linux environments.

At this stage, the implementation of ChatCompletion and Embedding has been completed.

**ChatCompletion** is adapted to  LLM  *phi3-mini*

**Samples**


1. download your LLM firstly and using pip to install python library


```bash

pip install -r requirement.txt

```

1. .env config your ChatCompletion  location

```txt

CHAT_COMPLETION_URL = 'Your chat completion model location'

```

2. Start your Local LLM Http Server

```bash

python local_llm_service.py

```

3. Add Microsoft.SemanticKernel, Microsoft.SemanticKernel.Connectors.AI.HuggingFace, Microsoft.SemanticKernel.Connectors.Memory.Qdrant(You can choose different vector database) packages 

4. Initialization endpoint for chatcompletion

```csharp

string chat_endpoint = "http://localhost:5002/v1/chat/completions";


```


5. Sample 1 - ChatCompletion


```csharp

using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Connectors.Memory.Qdrant;
using Microsoft.SemanticKernel.Plugins.Memory;
using Microsoft.SemanticKernel.Connectors.AI.HuggingFace.TextEmbedding;

#pragma warning disable SKEXP0020

Kernel kernel = new KernelBuilder()
            .AddHuggingFaceTextGeneration(
                model: "phi-3-mini",
                endpoint: chat_endpoint)
            .Build();

var questionAnswerFunction = kernel.CreateFunctionFromPrompt(@"{{$input}}");

var result = await kernel.InvokeAsync(questionAnswerFunction, new(){["input"] = "Can you introduce yourself?"});

result.GetValue<string>()


```









