import axios from "axios";

export async function openDoor() {
  const toolHost = "http://192.168.1.105:8000/tools/door";
  try {
    const response = await axios.post(toolHost);
    console.log(`Tool server returned ${response.data.byteLength} bytes`);
    return Buffer.from(response.data);
  } catch (error) {
    console.error(`Tool server error: ${error}`);
    return null;
  }
}

export const openDoorTool = {
  type: "function",
  function: {
    name: "openDoor",
    description:
      "Use this when the user asks you to open the door, or to let them in or out",
    parameters: {
      type: "object",
      required: [],
      properties: {},
    },
  },
};
