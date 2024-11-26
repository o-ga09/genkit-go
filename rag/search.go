package rag

import (
	"context"
	"log"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/dotprompt"
	"github.com/firebase/genkit/go/plugins/googleai"
	"github.com/firebase/genkit/go/plugins/localvec"
	"github.com/invopop/jsonschema"
)

const simpleQaPromptTemplate = `
You're a helpful agent that answers the user's common questions based on the context provided.

Here is the user's query: {{query}}

Here is the context you should use: {{context}}

Please provide the best answer you can.
`

type SimpleQAInput struct {
	Question string `json:"question"`
}

type SimpleQAPromptInput struct {
	Query   string `json:"query"`
	Context string `json:"context"`
}

func Search(ctx context.Context) error {
	// Intialize LLM API (Gemini)
	if err := googleai.Init(ctx, nil); err != nil {
		log.Fatal(err)
	}

	// ローカルベクトルストアを使用する
	// ローカルのファイルベースのベクトルストアは本番環境で使用しないこと
	err := localvec.Init()
	if err != nil {
		return err
	}

	// Select Model
	model := googleai.Model("gemini-1.5-flash")

	// Define embedder
	embedder := googleai.Embedder("text-embedding-004")

	// インデクサーとリトリーバーを定義する
	indexer, retriever, err := localvec.DefineIndexerAndRetriever(
		"simpleQA",
		localvec.Config{
			Embedder: embedder,
		},
	)
	if err != nil {
		log.Fatal(err)
	}

	simpleQAPrompt, err := dotprompt.Define("simpleQA",
		simpleQaPromptTemplate,
		dotprompt.Config{
			Model:        model,
			InputSchema:  jsonschema.Reflect(SimpleQAPromptInput{}),
			OutputFormat: ai.OutputFormatText,
		})
	if err != nil {
		return err
	}

	// 検索フローを定義する
	genkit.DefineFlow(
		"simpleQA",
		func(ctx context.Context, input *SimpleQAInput) (string, error) {
			d1 := ai.DocumentFromText("Paris is the capital of France", nil)
			d2 := ai.DocumentFromText("USA is the largest importer of coffee", nil)
			d3 := ai.DocumentFromText("Water exists in 3 states - solid, liquid and gas", nil)

			err := ai.Index(ctx, indexer, ai.WithIndexerDocs(d1, d2, d3))
			if err != nil {
				return "", err
			}

			dRequest := ai.DocumentFromText(input.Question, nil)
			response, err := ai.Retrieve(ctx, retriever, ai.WithRetrieverDoc(dRequest))
			if err != nil {
				return "", err
			}

			var sb strings.Builder
			for _, d := range response.Documents {
				sb.WriteString(d.Content[0].Text)
				sb.WriteByte('\n')
			}

			promptInput := &SimpleQAPromptInput{
				Query:   input.Question,
				Context: sb.String(),
			}

			resp, err := simpleQAPrompt.Generate(ctx,
				&dotprompt.PromptRequest{
					Variables: promptInput,
				},
				nil,
			)
			if err != nil {
				return "", err
			}
			return resp.Text(), nil
		})
	return nil
}
