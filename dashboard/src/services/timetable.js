import axios from 'axios'

const API_BASE = 'http://localhost:8000/api/sim/timetable'

const getTimetable = async () => {
    const response = await axios.get(API_BASE)
    return response.data
}

export default getTimetable
