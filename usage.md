  Candy Branch - Neural Data Processing Pipeline                                                          
                                                                                                          
  The candy branch provides advanced neural signal processing with two main features:                     
                                                                                                          
  1. Real-Time High Gamma Processing (NEW)                                                                
                                                                                                          
  Stream live neural data with real-time processing:                                                      
                                                                                                          
  # Terminal 1: Start the data stream                                                                     
  uv run brainstorm-stream --from-file data/hard/                                                         
                                                                                                          
  # Terminal 2: Start the real-time processor                                                             
  uv run python scripts/process_stream.py                                                                 
                                                                                                          
  # Terminal 3: Start the web viewer                                                                      
  uv run brainstorm-serve                                                                                 
                                                                                                          
  Then open http://localhost:8000 in your browser.                                                        
                                                                                                          
  The processor automatically:                                                                            
  - Detects and imputes 4 bad channels ([51, 232, 419, 582])                                              
  - Applies 70-150Hz bandpass filter (high gamma)                                                         
  - Extracts Hilbert envelope                                                                             
  - Normalizes with median + std                                                                          
  - Applies 3x3 spatial median filter                                                                     
  - Streams to WebSocket on port 8766                                                                     
                                                                                                          
  2. Offline Multi-Band Video Export                                                                      
                                                                                                          
  Generate MP4 videos with 4-panel frequency band visualization:                                          
                                                                                                          
  # Basic usage - generate deliverables                                                                   
  uv run python scripts/export_video.py -d super_easy -o /store1/candy/bci/super_easy.mp4 --verbose       
  uv run python scripts/export_video.py -d hard -o /store1/candy/bci/hard.mp4 --verbose                                                                
                                                                                                          
  Processing Pipeline:                                                                                    
  1. Bad channel detection (variance analysis on first 1000 frames)                                       
  2. Median interpolation from 8-connected neighbors (max 16 channels)                                    
  3. Artifact removal (60Hz notch + white noise reduction)                                                
  4. Frequency decomposition into 4 bands:                                                                
    - Theta/Alpha (4-12Hz) - movement planning                                                            
    - Beta (12-30Hz) - movement preparation                                                               
    - Low Gamma (30-70Hz) - local cortical processing                                                     
    - High Gamma (70-150Hz) - motor execution                                                             
  5. Per-band normalization                                                                               
  6. Maximum Intensity Projection (MIP) smoothing                                                         
  7. 2x2 panel visualization with colorbar and frame annotations     