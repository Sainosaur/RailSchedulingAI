import axios from 'axios';

const BASE_URL = 'http://localhost:8000/api/sim/lead';


const stallLead = async () => {
    const res = await axios.post(`${BASE_URL}/stall`);
    return res.data;
}

const releaseLead = async () => {
    const res = await axios.post(`${BASE_URL}/release`);
    return res.data;
}

const holdLead = async () => {
    const res = await axios.post(`${BASE_URL}/hold`);
    return res.data;
}


export { stallLead, releaseLead, holdLead };