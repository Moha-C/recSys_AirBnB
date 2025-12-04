```bash
git clone https://github.com/Moha-C/recSys_AirBnB recsys_startup
cd recsys_startup

python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

cd D:\recsys_startup_T\recsys_startup\trip-frontend
npm install        
npm run dev

cd recsys_startup
python run_all.py