package main

import (
	"context"
	"log"

	basicuse "github.com/o-ga09/genkit-go/basicUse"
	"github.com/o-ga09/genkit-go/rag"
)

func main() {
	ctx := context.Background()
	// basic flow use
	// basicuse.InitGenkit(ctx)

	// embededを実行
	// rag.Embeded(ctx)

	// RAGを実行
	if err := rag.Search(ctx); err != nil {
		log.Fatal(err)
	}

	// basic server use
	basicuse.BasicServer(ctx)
}
