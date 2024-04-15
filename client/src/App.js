import react ,{useEffect,useState} from 'react';

function App() {
  const [data,setData] = useState([]);
  useEffect(()=>{
    fetchData();
  })
  const fetchData = async () =>{
    try{const response = await fetch('http://localhost:5000/api/stock-data');
    const data = await response.json();
    setData(data);}catch(error){
        console.log(error);
    }
  }
  return (

    <div className="App">
      
    </div>
  );
}

export default App;
