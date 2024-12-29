import { Client, Message, OmitPartialGroupDMChannel, TextChannel } from "discord.js";
import { CustomEventEmitter } from "../utils/custom-event-emitter.js";
import { Llama } from "../llama/llama.js";

type DiscordBotEvents = {
    botStarted: {};
};

export class DiscordBot extends CustomEventEmitter<DiscordBotEvents> {
    private client: Client;
    private token: string;
    private clientId: string;

    private llama: Llama;
    private activeChannelId?: string;
    private typingDebounceTimer?: NodeJS.Timeout;

    constructor(token: string, clientId: string) {
        super();
        this.token = token;
        this.clientId = clientId;
        this.client = new Client({ intents: [(0x3317EFD | 0x2 | 0x8000) & ~(65536 | 64)] }); // ALL_UNPRIVILEGED + GUILD_MEMBERS + MESSAGE_CONTENT - GUILD_SCHEDULED_EVENTS - GUILD_INVITES
        this.llama = new Llama('discordTesting');

        this.sendIsTypingToActiveChannel = this.sendIsTypingToActiveChannel.bind(this);
        this.onMessageCreated = this.onMessageCreated.bind(this);

        this.llama.on('messageInProgress', this.sendIsTypingToActiveChannel)
    }

    private async sendIsTypingToActiveChannel() {
        if (this.activeChannelId) {
            const channelId = this.activeChannelId;
            clearTimeout(this.typingDebounceTimer);

            this.typingDebounceTimer = setTimeout(async () => {
                const channel = this.client.channels.cache.get(channelId);
                await (channel as TextChannel).sendTyping();
            }, 8000);
        }
    }

    private async onMessageCreated(message: OmitPartialGroupDMChannel<Message<boolean>>) {
        if (message.author.id === this.client.user?.id) {
            return;
        }

        if (message.channelId !== '1322487628120985640') {
            return;
        }

        console.log(message.content);

        this.activeChannelId = message.channelId;

        const reply = await this.llama.runPrompt(message.content, message.author.displayName);

        await this.sendToChannel(reply, message.channelId);
    }

    public async start() {
        // Alert when logged in
        this.client.on('ready', () => {
            console.log(`Logged in as ${this.client.user?.tag}!`);
            this.emit('botStarted', {});
        });

        this.client.on('messageCreate', this.onMessageCreated);

        this.client.login(this.token);
    }

    private async sendToChannel(text: string, channelId: string) {
        const channel = this.client.channels.cache.get(channelId);
        await (channel as TextChannel).send(text);
    }
}