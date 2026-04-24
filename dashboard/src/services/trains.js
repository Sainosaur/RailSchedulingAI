import axios from "axios";

const BASE_URL = "http://localhost:8000/api/dashboard/trains";

const getAll = async () => {
  const response = await axios.get(BASE_URL);
  return response.data;
};

export default getAll;
