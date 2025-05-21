import axios from "axios";
import { Llama, MemoryModel } from "../llama/llama.js";

const MODEL = "llama3.3";

export type PromptRequest = {
  prompt: string;
  userId: string;

  personality?: string;
  gender?: string;
  sourceMaterial?: string;
};

export type PromptResponse = {
  respondingTo: string;
  response: string;
  audio?: string;
};

export type StatusResponse = {
  status: number;
  message: string;
};

export type HistoryRequest = {
  limit: number;
  userId: string;
};

export type HistoryResponse = {
  messages: MemoryModel[];
};

export type DeleteHistoryRequest = {
  userId: string;
};

type ApiLlama = {
  userId: string;
  llama: Llama;
  isListeningMode: boolean;
};

export class GenericApi {
  private readonly llamas: ApiLlama[];

  constructor() {
    this.llamas = [];

    this.handleRequest = this.handleRequest.bind(this);
    this.handleVoiceRequest = this.handleVoiceRequest.bind(this);
    this.handleHistoryRequest = this.handleHistoryRequest.bind(this);
    this.handleDeleteHistoryRequest =
      this.handleDeleteHistoryRequest.bind(this);
    this.runOllama = this.runOllama.bind(this);
    this.runFishSpeech = this.runFishSpeech.bind(this);
    this.runRVC = this.runRVC.bind(this);
  }

  private async onLlamaPartRecieved(
    prompt: string,
    personality: string,
    text: string,
    accumulatedMessage: string,
    onChunkUpdate?: (chunk: PromptResponse) => void
  ) {
    // If there's no callback waiting for streamed content, we can skip all this
    if (!onChunkUpdate) {
      return accumulatedMessage;
    }

    accumulatedMessage += text;

    // Test if we've accumulated a new sentence, and if so, generate audio for it
    const splitSentences = accumulatedMessage
      .replace(/([.?!])\s*(?=[A-Z])/g, "$1|")
      .split("|");
    let audio: string | undefined = undefined;
    if (splitSentences.length == 2) {
      accumulatedMessage = accumulatedMessage
        .substring(splitSentences[0].length)
        .trimStart();

      audio = (
        (await this.convertTextToAudioAndGetResponse(
          prompt,
          splitSentences[0],
          personality ?? "Rimuru"
        )) as PromptResponse
      ).audio;
    }

    // Note: !text seems to indicate the end of generation
    if (!text) {
      audio = (
        (await this.convertTextToAudioAndGetResponse(
          prompt,
          accumulatedMessage,
          personality ?? "Rimuru"
        )) as PromptResponse
      ).audio;
    }

    onChunkUpdate({
      respondingTo: prompt,
      response: text,
      audio,
    });

    return accumulatedMessage;
  }

  public async handleRequest(
    request: PromptRequest,
    onChunkUpdate?: (chunk: PromptResponse) => void
  ): Promise<PromptResponse | StatusResponse> {
    const { prompt, userId, personality, gender, sourceMaterial } = request;
    if (!prompt || !userId) {
      return { status: 400, message: "Prompt and userId are required" };
    }

    let accumulatedMessage = "";

    const ollamaOutput = await this.runOllama(
      prompt,
      userId,
      personality,
      gender,
      sourceMaterial,
      undefined, // leave default value
      undefined, // leave default value
      async (text) => {
        accumulatedMessage = await this.onLlamaPartRecieved(
          prompt,
          personality ?? "Rimuru",
          text,
          accumulatedMessage,
          onChunkUpdate
        );
      }
    );
    if (!ollamaOutput) {
      return { status: 500, message: "Ollama processing failed." };
    }

    let ollamaOutputWithoutThink = ollamaOutput;
    if (ollamaOutput.includes("</think>")) {
      ollamaOutputWithoutThink = ollamaOutput.substring(
        ollamaOutput.indexOf("</think>") + "</think>".length
      );
    }

    return onChunkUpdate
      ? { respondingTo: prompt, response: "" }
      : await this.convertTextToAudioAndGetResponse(
          prompt,
          ollamaOutputWithoutThink,
          personality ?? "Rimuru"
        );
  }

  public async handleVoiceRequest(
    request: PromptRequest,
    onChunkUpdate?: (chunk: PromptResponse) => void
  ): Promise<PromptResponse | StatusResponse> {
    const { prompt, userId, personality, gender, sourceMaterial } = request;
    if (!prompt || !userId) {
      return { status: 400, message: "Prompt and userId are required" };
    }

    const filteredPrompt = this.filterWordsFromSTT(prompt);

    const llama = this.createLlama(userId, personality, gender, sourceMaterial);

    // Before anything, check for needed tools.
    // Note: If this too often calls tools by accident,
    // we may need to do it either *in* or after shouldRespond
    await llama.llama.useTools(prompt);

    const shouldRespond =
      llama.isListeningMode ||
      (await llama.llama.shouldRespond(
        filteredPrompt,
        userId,
        personality ?? "Rimuru",
        Llama.getShouldRespondForPersonality(
          personality ?? "Rimuru",
          gender ?? "male",
          sourceMaterial ?? "That Time I got Reincarnated as a Slime"
        )
      ));

    // For now, we're only saving prompts rimuru responds to
    // await llama.llama.saveIncomingPrompt(filteredPrompt);

    if (shouldRespond) {
      llama.isListeningMode = true;
      await llama.llama.saveIncomingPrompt(filteredPrompt, userId);

      let accumulatedMessage = "";

      const ollamaOutput = await this.runOllama(
        filteredPrompt,
        userId,
        personality,
        gender,
        sourceMaterial,
        false,
        false,
        async (text) => {
          accumulatedMessage = await this.onLlamaPartRecieved(
            filteredPrompt,
            personality ?? "Rimuru",
            text,
            accumulatedMessage,
            onChunkUpdate
          );
        }
      );
      if (!ollamaOutput) {
        return { status: 500, message: "Ollama processing failed." };
      }

      let ollamaOutputWithoutThink = ollamaOutput;
      if (ollamaOutput.includes("</think>")) {
        ollamaOutputWithoutThink = ollamaOutput.substring(
          ollamaOutput.indexOf("</think>") + "</think>".length
        );
      }

      // At this point I'm really just assuming we're using streaming.
      // If we are, we can take a little time at the end here to do convoEnd
      // processing, and it'll likely happen before the user notices.
      // Basically if we think the user is done talking to us, we'll go back
      // to strictly waiting to be addressed.
      llama.isListeningMode = !(await llama.llama.isConvoEnd(
        filteredPrompt,
        userId,
        Llama.getConvoEndForPersonality(
          personality ?? "Rimuru",
          gender ?? "male",
          sourceMaterial ?? "That Time I got Reincarnated as a Slime"
        )
      ));

      return onChunkUpdate
        ? { respondingTo: filteredPrompt, response: "" }
        : await this.convertTextToAudioAndGetResponse(
            filteredPrompt,
            ollamaOutputWithoutThink,
            personality ?? "Rimuru"
          );
    } else {
      return { status: 204, message: "Ollama determined not to respond" };
    }
  }

  public async handleHistoryRequest(
    request: HistoryRequest
  ): Promise<HistoryResponse | StatusResponse> {
    const { limit, userId } = request;
    if (!limit || !userId) {
      return { status: 400, message: "Limit and userId are required" };
    }

    const llama = this.createLlama(userId);
    if (llama) {
      // We don't want this llama to stick around, just in case the user wasn't done typing...
      // Moon's note: this is nasty, but I'll remnid you this is only a debug interface
      this.deleteLlama(userId);
      return {
        messages: await llama.llama.getChatHistory(limit),
      };
    }

    return { status: 404, message: "That Llama does not exist" };
  }

  public async handleDeleteHistoryRequest(
    request: DeleteHistoryRequest
  ): Promise<StatusResponse> {
    const { userId } = request;
    if (!userId) {
      return { status: 400, message: "UserId is required" };
    }

    const llama = this.createLlama(userId);
    if (llama) {
      await llama.llama.deleteConversation();
      this.deleteLlama(userId);

      return { status: 200, message: "Conversation was deleted" };
    }

    return { status: 404, message: "That Llama does not exist" };
  }

  private createLlama(
    userId: string,
    personality: string = "Rimuru",
    gender: string = "male",
    sourceMaterial: string = "That Time I got Reincarnated as a Slime"
  ) {
    let llama = this.llamas.find((x) => x.userId === userId);
    if (!llama) {
      // Just for arthur
      let user = userId;
      if (user === "moon1945") {
        user = "moon";
      } else if (user === "moon") {
        user = "viyi";
      }

      llama = {
        llama: new Llama(
          user,
          MODEL,
          Llama.getSystemPromptForPersonality(
            personality,
            gender,
            sourceMaterial
          )
        ),
        userId,
        isListeningMode: false,
      };
      this.llamas.push(llama);
    }
    return llama;
  }

  private getLlama(userId: string) {
    return this.llamas.find((x) => x.userId === userId);
  }

  private deleteLlama(userId: string) {
    const index = this.llamas.findIndex((x) => x.userId === userId);
    if (index > -1) {
      this.llamas.splice(index, 1);
    }
  }

  private filterWordsForTTS(text: string) {
    return text
      .replace("Rimuru", "Reemaru")
      .replace("Shion", "Sheeown")
      .replace("*", "")
      .replace(" - ", ", ");
  }

  private filterWordsFromSTT(text: string) {
    return text
      .replace(" Reamer", " Rimuru")
      .replace(" reamer", " Rimuru")
      .replace(" Rimmer", " Rimuru")
      .replace(" rimmer", " Rimuru")
      .replace(" Reimuer", " Rimuru")
      .replace(" reimuer", " Rimuru")
      .replace(" Remaru", " Rimuru")
      .replace(" remaru", " Rimuru")
      .replace(" Remerow", " Rimuru")
      .replace(" remerow", " Rimuru")
      .replace(" Reemaru", " Rimuru")
      .replace(" reemaru", " Rimuru")
      .replace(" Reemuru", " Rimuru")
      .replace(" reemuru", " Rimuru")
      .replace(" Rimaru", " Rimuru")
      .replace(" rimaru", " Rimuru")
      .replace(" Remeru", " Rimuru")
      .replace(" remeru", " Rimuru")
      .replace(" Remer", " Rimuru")
      .replace(" remer", " Rimuru")
      .replace(" Imaru", " Rimuru")
      .replace(" imaru", " Rimuru")
      .replace(" Remaroo", " Rimuru")
      .replace(" remaroo", " Rimuru");
  }

  private async convertTextToAudioAndGetResponse(
    respondingTo: string,
    response: string,
    personality: string
  ): Promise<PromptResponse | StatusResponse> {
    // Right now we only support Rimuru and Frieren voices
    if (
      personality !== "Rimuru" &&
      personality !== "Frieren" &&
      personality !== "Gura"
    ) {
      personality = "default";
    }

    const fishOutput = await this.runFishSpeech(
      this.filterWordsForTTS(response),
      personality
    );
    if (!fishOutput) {
      return { status: 500, message: "Fish-speech processing failed." };
    }

    let finalOutput = fishOutput;
    if (personality !== "default") {
      const rvcOutput = await this.runRVC(fishOutput, personality);
      if (!rvcOutput) {
        return { status: 500, message: "RVC processing failed." };
      }

      finalOutput = rvcOutput;
    }

    return {
      respondingTo,
      response,
      audio: finalOutput.toString("base64"),
    };
  }

  private async runOllama(
    prompt: string,
    userId: string,
    personality: string = "Rimuru",
    gender: string = "male",
    sourceMaterial: string = "That Time I got Reincarnated as a Slime",
    saveIncomingPrompt: boolean = true,
    useTools: boolean = true,
    onChunkUpdate?: (text: string) => Promise<void>
  ): Promise<string | null> {
    try {
      const llama = this.createLlama(
        userId,
        personality,
        gender,
        sourceMaterial
      );

      if (useTools) {
        // console.log("Using tools...");
        // await llama.llama.useTools(prompt);
        // console.log("Used.");
      }

      if (saveIncomingPrompt) {
        await llama.llama.saveIncomingPrompt(prompt, userId);
      }

      return await llama.llama.runPrompt(prompt, userId, onChunkUpdate);
    } catch (error) {
      console.error(`Ollama error: ${error}`);
      return null;
    }
  }

  private async runFishSpeech(
    text: string,
    personality: string,
    stream?: boolean,
    audioChunkRecieved?: (audio: Buffer) => void
  ): Promise<Buffer | null> {
    const ttsHost = "http://192.168.1.103:8080/v1/tts";
    const payload = {
      text,
      format: "wav",
      reference_id: personality.toLowerCase(),
      use_memory_cache: "on",
      normalize: "false",
      streaming: !!stream,
    };
    try {
      const response = await axios.post(ttsHost, payload, {
        headers: { accept: "*/*", "Content-Type": "application/json" },
        responseType: stream ? "stream" : "arraybuffer",
      });

      if (stream) {
        const stream = response.data;

        console.log(`Streaming Fish-speech response...`);

        for await (const chunk of stream) {
          audioChunkRecieved!(Buffer.from(chunk));
        }
        return null;
      } else {
        console.log(`Fish-speech returned ${response.data.byteLength} bytes`);
        return Buffer.from(response.data);
      }
    } catch (error) {
      console.error(`Fish-speech error: ${error}`);
      return null;
    }
  }

  private async runRVC(
    data: Buffer,
    personality: string
  ): Promise<Buffer | null> {
    const rvcHost = "http://192.168.1.107:8080";
    const audioData = data.toString("base64");
    const payload = { audio_data: audioData };
    try {
      // Load the correct RVC model
      await axios.post(`${rvcHost}/models/${personality.toLowerCase()}`, {
        headers: { "Content-Type": "application/json" },
        responseType: "arraybuffer",
      });
      console.log(`RVC loaded model: ${personality.toLowerCase()}`);

      const response = await axios.post(`${rvcHost}/convert`, payload, {
        headers: { "Content-Type": "application/json" },
        responseType: "arraybuffer",
      });
      console.log(`RVC returned ${response.data.byteLength} bytes`);
      return Buffer.from(response.data);
    } catch (error) {
      console.error(`RVC error: ${error}`);
      return null;
    }
  }
}
