import express, { Express, Request, Response } from "express";
import {
  StatusResponse,
  GenericApi,
  HistoryRequest,
  HistoryResponse,
  PromptRequest,
  PromptResponse,
} from "./genericApi.js";

export class ApiServer {
  private readonly port: number;
  private readonly express: Express;
  private readonly genericApi: GenericApi;

  constructor(port: number = 8080) {
    this.port = port;
    this.express = express();
    this.express.use(express.json());

    this.genericApi = new GenericApi();

    this.handleRequest = this.handleRequest.bind(this);
    this.handleVoiceRequest = this.handleVoiceRequest.bind(this);
    this.handleHistoryRequest = this.handleHistoryRequest.bind(this);
    this.handleDeleteHistoryRequest =
      this.handleDeleteHistoryRequest.bind(this);

    this.express.post("/process", this.handleRequest);
    this.express.post("/processVoice", this.handleVoiceRequest);
    this.express.post("/getHistory", this.handleHistoryRequest);
    this.express.post("/deleteHistory", this.handleDeleteHistoryRequest);
  }

  public start() {
    this.express.listen(this.port, () => {
      console.log(`Server running on port ${this.port}`);
    });
  }

  private sendResponseChunk(response: PromptResponse, res: Response) {
    res.write(
      JSON.stringify({
        respondingTo: response.respondingTo,
        response: response.response,
        audio: response.audio,
      }) + "\n" // The delimiter is important! It lets the client cleanly consume jumbled chunks of responses
    );
  }

  private async handleRequest(req: Request, res: Response) {
    res.setHeader("Content-Type", "application/json");
    res.setHeader("Transfer-Encoding", "chunked");

    const response = await this.genericApi.handleRequest(
      req.body as PromptRequest,
      (response) => this.sendResponseChunk(response, res)
    );

    // We assume it's an error if the message property exists...
    // Maybe change this? But for now it's fine
    const errorResponse = response as StatusResponse;
    if (errorResponse.message) {
      res.status(errorResponse.status).json({ detail: errorResponse.message });
    } else {
      res.end();
    }
  }

  private async handleVoiceRequest(req: Request, res: Response) {
    res.setHeader("Content-Type", "application/json");
    res.setHeader("Transfer-Encoding", "chunked");

    const response = await this.genericApi.handleVoiceRequest(
      req.body as PromptRequest,
      (response) => this.sendResponseChunk(response, res)
    );

    // We assume it's an error if the message property exists...
    // Maybe change this? But for now it's fine
    const errorResponse = response as StatusResponse;
    if (errorResponse.message) {
      res.status(errorResponse.status).json({ detail: errorResponse.message });
    } else {
      res.end();
    }
  }

  private async handleHistoryRequest(req: Request, res: Response) {
    const response = await this.genericApi.handleHistoryRequest(
      req.body as HistoryRequest
    );

    const successResponse = response as HistoryResponse;

    // We assume it's an error if the message property exists...
    // Maybe change this? But for now it's fine
    const errorResponse = response as StatusResponse;
    if (errorResponse.message) {
      res.status(errorResponse.status).json({ detail: errorResponse.message });
    } else {
      res.json([...successResponse.messages]);
    }
  }

  private async handleDeleteHistoryRequest(req: Request, res: Response) {
    const response = await this.genericApi.handleDeleteHistoryRequest(
      req.body as HistoryRequest
    );

    res.status(response.status).json({ detail: response.message });
  }
}
