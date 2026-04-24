import axios from "axios";

const BASE_URL = "http://localhost:8000/api/dashboard/kill";

const getKillStatus = async () => {
  const response = await axios.get(BASE_URL);
  return response.data;
};

const killServer = async () => {
  await axios.post(`${BASE_URL}/kill`);
};

const restoreServer = async () => {
  await axios.post(`${BASE_URL}/restore`);
};

export { killServer, restoreServer, getKillStatus };
