package basicuse

import (
	"context"
	"log"
	"log/slog"
	"net/http"
	"os"

	"github.com/firebase/genkit/go/genkit"
)

func BasicServer(ctx context.Context) {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{}))
	slog.SetDefault(logger)

	// genkit.NewFlowServeMuxにpathを渡すと、エンドポイントとして登録される
	path := []string{
		"summary",
		"menuQA",
		"simpleQA",
	}

	// チャットフローを呼び出す
	// if err := aichat.Chat(ctx); err != nil {
	// 	log.Fatal(err)
	// }
	// gnkitでAPIサーバーにフローを登録する
	mux := genkit.NewFlowServeMux(path)

	// APIサーバーを起動する
	if err := http.ListenAndServe(":8080", mux); err != nil {
		log.Fatal(err)
	}
}
