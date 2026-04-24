import axios from 'axios';


const BASE_URL = 'http://localhost:8000/api/sim';

const startSimulation = async () => {
    const res = await axios.post(`${BASE_URL}/start`);
    return res.data;
}

const stopSimulation = async () => {
    const res = await axios.post(`${BASE_URL}/stop`);
    return res.data;
}

const simStatus = async () => {
    const res = await axios.get(`${BASE_URL}/status`);
    return res.data;
}

export { startSimulation, stopSimulation, simStatus };