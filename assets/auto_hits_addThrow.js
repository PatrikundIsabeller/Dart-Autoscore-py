// auto_hits_addThrow.js
(function(){
  const API_URL   = window.TRIPLEONE_API || 'http://127.0.0.1:8017';
  const CAMERA_ID = window.TRIPLEONE_CAMERA || '';
  const POLL_MS   = 300;
  function hitToObj(hit){
    const map = { single:'S', double:'D', treble:'T', outer_bull:'S', inner_bull:'D', miss:'M' };
    let ring = map[hit.ring] || 'S';
    let n = (hit.ring==='outer_bull' || hit.ring==='inner_bull') ? 25 : (hit.sector || 0);
    let value = hit.score|0;
    let label;
    if(ring==='M'){ n=0; value=0; label='0'; }
    else if(n===25 && ring==='D'){ label='D25'; }
    else if(n===25 && ring==='S'){ label='25'; }
    else { label = `${ring}${n}`; }
    return { ring, n, value, label };
  }
  async function pollOnce(){
    try{
      const url = new URL(API_URL + '/hits');
      url.searchParams.set('clear','1');
      url.searchParams.set('max_items','16');
      if(CAMERA_ID) url.searchParams.set('camera_id', CAMERA_ID);
      const res = await fetch(url.toString(), {cache:'no-store'});
      const data = await res.json();
      if(Array.isArray(data.hits) && typeof window.addThrow === 'function'){
        for(const hit of data.hits){
          const obj = hitToObj(hit);
          window.addThrow(obj);
        }
      }
    }catch(e){ /* silent */ }
  }
  setInterval(pollOnce, POLL_MS);
})();
