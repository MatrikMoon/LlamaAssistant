import { DiscordBot } from "./discord/discordService.js";
import { config } from "./config.js";
import { ApiServer } from "./api/restApi.js";

// const discordBot = new DiscordBot(config.DISCORD_TOKEN, config.DISCORD_CLIENT_ID);
// await discordBot.start();

const api = new ApiServer(8080);
api.start();
