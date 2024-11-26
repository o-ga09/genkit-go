package rag

import (
	"context"
	"io"
	"log"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googleai"
	"github.com/firebase/genkit/go/plugins/localvec"
	"github.com/ledongthuc/pdf"
	"github.com/tmc/langchaingo/textsplitter"
)

func Embeded(ctx context.Context) error {
	// Intialize LLM API (Gemini)
	if err := googleai.Init(ctx, nil); err != nil {
		log.Fatal(err)
	}

	// ローカルベクトルストアを使用する
	err := localvec.Init()
	if err != nil {
		return err
	}

	// インデクサーを作成する
	menuPDFIndexer, _, err := localvec.DefineIndexerAndRetriever(
		"menuQA",
		localvec.Config{
			Embedder: googleai.Embedder("text-embedding-004"),
		},
	)
	if err != nil {
		return err
	}

	// チャンク分割する
	splitter := textsplitter.NewRecursiveCharacter(
		textsplitter.WithChunkSize(200),
		textsplitter.WithChunkOverlap(20),
	)

	// インデクサーフローを定義する
	genkit.DefineFlow(
		"indexMenu",
		func(ctx context.Context, path string) (any, error) {
			// Extract plain text from the PDF. Wrap the logic in Run so it
			// appears as a step in your traces.
			pdfText, err := genkit.Run(ctx, "extract", func() (string, error) {
				return readPDF(path)
			})
			if err != nil {
				return nil, err
			}

			// Split the text into chunks. Wrap the logic in Run so it
			// appears as a step in your traces.
			docs, err := genkit.Run(ctx, "chunk", func() ([]*ai.Document, error) {
				chunks, err := splitter.SplitText(pdfText)
				if err != nil {
					return nil, err
				}

				var docs []*ai.Document
				for _, chunk := range chunks {
					docs = append(docs, ai.DocumentFromText(chunk, nil))
				}
				return docs, nil
			})
			if err != nil {
				return nil, err
			}

			// Add chunks to the index.
			err = ai.Index(ctx, menuPDFIndexer, ai.WithIndexerDocs(docs...))
			return nil, err
		},
	)
	return nil
}

// Helper function to extract plain text from a PDF. Excerpted from
// https://github.com/ledongthuc/pdf
func readPDF(path string) (string, error) {
	f, r, err := pdf.Open(path)
	if f != nil {
		defer f.Close()
	}
	if err != nil {
		return "", err
	}

	reader, err := r.GetPlainText()
	if err != nil {
		return "", err
	}

	bytes, err := io.ReadAll(reader)
	if err != nil {
		return "", err
	}
	return string(bytes), nil
}
