package basicuse

import (
	"context"
	"errors"
	"fmt"
	"log"

	// Import the Genkit core libraries.
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"

	// Import the Google AI plugin.
	"github.com/firebase/genkit/go/plugins/googleai"
)

func InitGenkit(ctx context.Context) {
	// Initialize the Google AI plugin. When you pass an empty string for the
	// apiKey parameter, the Google AI plugin will use the value from the
	// GOOGLE_GENAI_API_KEY environment variable, which is the recommended
	// practice.
	if err := googleai.Init(ctx, nil); err != nil {
		log.Fatal(err)
	}

	// Define a simple flow that prompts an LLM to generate menu suggestions.
	flow := basicFlow(ctx)

	fmt.Println(flow.Name())
	res, err := flow.Run(ctx, "Japanese")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(res)
}

func basicFlow(ctx context.Context) *genkit.Flow[string, string, struct{}] {
	return genkit.DefineFlow("menuSuggestionFlow", func(ctx context.Context, input string) (string, error) {
		// The Google AI API provides access to several generative models. Here,
		// we specify gemini-1.5-flash.
		m := googleai.Model("gemini-1.5-flash")
		if m == nil {
			return "", errors.New("menuSuggestionFlow: failed to find model")
		}

		// Construct a request and send it to the model API.
		resp, err := m.Generate(ctx,
			ai.NewGenerateRequest(
				&ai.GenerationCommonConfig{Temperature: 1},
				ai.NewUserTextMessage(fmt.Sprintf(`Suggest an item for the menu of a %s themed restaurant`, input)),
			),
			nil)
		if err != nil {
			return "", err
		}
		// Handle the response from the model API. In this sample, we just
		// convert it to a string, but more complicated flows might coerce the
		// response into structured output or chain the response into another
		// LLM call, etc.
		text := resp.Text()
		return text, nil
	})
}
