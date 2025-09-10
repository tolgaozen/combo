package internal

import (
	"context"
	"fmt"

	"github.com/sashabaranov/go-openai"
)

// SupportedModels defines the list of supported OpenAI models
const (
	ModelGPT35Turbo = "gpt-3.5-turbo"
	ModelGPT4oMini  = "gpt-4o-mini"
	ModelGPT4o      = "gpt-4o"
	ModelGPT4Turbo  = "gpt-4-turbo"
	ModelGPT4       = "gpt-4"
)

// SupportedModelsList contains all supported models as a slice
var SupportedModelsList = []string{
	ModelGPT35Turbo,
	ModelGPT4oMini,
	ModelGPT4o,
	ModelGPT4Turbo,
	ModelGPT4,
}

// OpenAIClient wraps the OpenAI client
type OpenAIClient struct {
	client *openai.Client
	model  string
}

// NewOpenAIClient initializes a new OpenAIClient with default model
func NewOpenAIClient(apiKey string) *OpenAIClient {
	return &OpenAIClient{
		client: openai.NewClient(apiKey),
		model:  ModelGPT4oMini, // Default model
	}
}

// NewOpenAIClientWithModel initializes a new OpenAIClient with specified model
func NewOpenAIClientWithModel(apiKey, model string) (*OpenAIClient, error) {
	if !isValidModel(model) {
		return nil, fmt.Errorf("unsupported model: %s. Supported models: %v", model, SupportedModelsList)
	}

	return &OpenAIClient{
		client: openai.NewClient(apiKey),
		model:  model,
	}, nil
}

// SetModel updates the model for the client
func (o *OpenAIClient) SetModel(model string) error {
	if !isValidModel(model) {
		return fmt.Errorf("unsupported model: %s. Supported models: %v", model, SupportedModelsList)
	}
	o.model = model
	return nil
}

// GetModel returns the current model
func (o *OpenAIClient) GetModel() string {
	return o.model
}

// isValidModel checks if the model is supported
func isValidModel(model string) bool {
	for _, supportedModel := range SupportedModelsList {
		if model == supportedModel {
			return true
		}
	}
	return false
}

// CreateChatCompletionRequest constructs the ChatCompletionRequest with the client's model
func (o *OpenAIClient) CreateChatCompletionRequest(prompt, diff string) openai.ChatCompletionRequest {
	return openai.ChatCompletionRequest{
		Model: o.model,
		Messages: []openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleSystem,
				Content: prompt,
			},
			{
				Role:    openai.ChatMessageRoleUser,
				Content: diff,
			},
		},
		Temperature:      0.7,
		TopP:             1,
		FrequencyPenalty: 0,
		PresencePenalty:  0,
		MaxTokens:        200,
		N:                1,
		Stream:           false,
	}
}

// SendChatCompletionRequest sends the request and returns the response
func (o *OpenAIClient) SendChatCompletionRequest(ctx context.Context, request openai.ChatCompletionRequest) (openai.ChatCompletionResponse, error) {
	return o.client.CreateChatCompletion(ctx, request)
}
