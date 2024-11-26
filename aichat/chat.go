package aichat

import (
	"context"
	"log"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/dotprompt"
	"github.com/firebase/genkit/go/plugins/googleai"
	"github.com/invopop/jsonschema"
)

type PromptInput struct {
	URL string `json:"url"`
}

func Chat(ctx context.Context) error {
	// Intialize LLM API (Gemini)
	if err := googleai.Init(ctx, nil); err != nil {
		log.Fatal(err)
	}

	// Select Model
	model := googleai.Model("gemini-1.5-flash")

	// プロンプトを定義
	prompt, err := dotprompt.Define("AI Chat Assistant",
		"First, fetch this link: {{url}}. Then, summarize the content within 20 words.",
		dotprompt.Config{
			Model: model,
			// Tools: ,
			GenerationConfig: &ai.GenerationCommonConfig{
				Temperature: 0.7,
			},
			InputSchema:  jsonschema.Reflect(PromptInput{}),
			OutputFormat: ai.OutputFormatText,
		},
	)
	if err != nil {
		return err
	}

	// genkitのフローを定義
	genkit.DefineFlow("summary", func(ctx context.Context, input string) (string, error) {
		resp, err := prompt.Generate(ctx,
			&dotprompt.PromptRequest{
				Variables: &PromptInput{
					URL: input,
				},
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
