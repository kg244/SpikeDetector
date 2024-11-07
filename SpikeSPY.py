#Latest code as of 10.29.24
#------------------------------------------------------------------------------------------------------------
import bioread
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io
import xlwings as xw

main_dir = os.path.dirname(os.path.abspath(__file__))

def SpikeSpy(file_name, test, datafolder, low_freq, high_freq, threshold, reset_threshold, graph_downsample, master_df, window_size, slide_size):
	for pos, data in enumerate(test.channels):
		data = test.channels[pos].data
		# Define the sampling rate and convert window and slide sizes to data points
		sampling_rate = int(test.samples_per_second)  # samples per second
		window_size_samples = window_size * sampling_rate
		slide_size_samples = slide_size * sampling_rate

		# Initialize tracking variables for seizure events
		seizure_event = 0
		master_peaks = [] #not useful yet--> factor in downsampling
		master_peak_val = []
		starts = []
		ends = []
		ongoing_seizure = False
		current_start = None
		current_end = None
		start = 0

		# Check if threshold is negative or positive
		polarity_is_positive = threshold > 0

		while start + window_size_samples <= len(data):
			end = start + window_size_samples
			window = data[int(start):int(end)]

			peak_starts = []
			temp = []
			start_flag = 0
			reset_flag = False

			for i in window:
				sample_index = start + 1
				sample_time = sample_index / sampling_rate

				if polarity_is_positive:
					# Handle positive polarity (standard peak detection)
					if start_flag == 0 and i >= threshold:
						start_flag = 1
						reset_flag = False
						temp.append(sample_time)
					elif start_flag == 1 and i >= threshold:
						temp.append(sample_time)
					elif start_flag == 1 and i < threshold and i >= reset_threshold:
						temp.append(sample_time)
					elif start_flag == 1 and i < reset_threshold:
						temp.append(sample_time)
						start_flag = 0
						reset_flag = True
						peak_starts.append(np.mean(temp))
						temp = []

				else:
					# Handle negative polarity (inverted peak detection)
					if start_flag == 0 and i <= threshold:  # Detecting below negative threshold
						start_flag = 1
						reset_flag = False
						temp.append(sample_time)
					elif start_flag == 1 and i <= threshold:
						temp.append(sample_time)
					elif start_flag == 1 and i > threshold and i <= reset_threshold:
						temp.append(sample_time)
					elif start_flag == 1 and i > reset_threshold:
						temp.append(sample_time)
						start_flag = 0
						reset_flag = True
						peak_starts.append(np.mean(temp))
						temp = []

			start += slide_size_samples  # Increment by the slide size

			# If the frequency of peaks matches the criteria, log the seizure event
			if (low_freq * window_size) <= len(peak_starts) <= (high_freq * window_size):
				# Current event start and end
				event_start = start / sampling_rate
				event_end = end / sampling_rate

				# Check for overlap or immediate succession with the ongoing event
				if ongoing_seizure and event_start <= current_end + slide_size:
					# Extend the current event's end time if it overlaps or is adjacent
					current_end = max(current_end, event_end)
				else:
					# Finalize the previous event if there is one
					if ongoing_seizure:
						starts.append(current_start)
						ends.append(current_end)

					# Start a new event
					current_start = event_start
					current_end = event_end
					ongoing_seizure = True

		# After the loop, finalize any ongoing event
		if ongoing_seizure:
			starts.append(current_start)
			ends.append(current_end)

		# Merging consecutive or overlapping events
		merged_starts = []
		merged_ends = []
		if starts and ends:
			current_start = starts[0]
			current_end = ends[0]

			for start, end in zip(starts[1:], ends[1:]):
				# Check if the current event is continuous with the previous one
				if start <= current_end + slide_size:
					# Extend the current event
					current_end = max(current_end, end)
				else:
					# Finalize the current event and start a new one
					merged_starts.append(current_start)
					merged_ends.append(current_end)
					current_start = start
					current_end = end

			# Append the last merged event
			merged_starts.append(current_start)
			merged_ends.append(current_end)

		# Downsample the data for visualization
		timer = [pos / sampling_rate for pos, i in enumerate(data)]
		data_builder = {'Data': data, 'Time (seconds)': timer}
		datadf = pd.DataFrame(data_builder)
		datadf2 = datadf[::graph_downsample]

		peak_maker = [threshold for t in master_peaks]
		peak_builder = {'Data': peak_maker, 'Time (seconds)': master_peaks}
		peakdf = pd.DataFrame(peak_builder)

		fig_name = file_name.split('.')[0]
		fig_nammer = f'Trace from {fig_name}_Ch.{pos+1}'

		fig = go.Figure()
		fig.add_trace(go.Scatter(x=datadf2['Time (seconds)'], y=datadf2['Data'], mode='lines', name=f'Data (downsampled {graph_downsample}x)'))
		fig.add_trace(go.Scatter(x=peakdf['Time (seconds)'], y=peakdf['Data'], mode='markers', marker={'color': 'lightsalmon'}, name='Seizure Activity'))
		fig.add_trace(go.Scatter(x=list(range(0, int(datadf2['Time (seconds)'].values.max()), 5)), y=[threshold for i in list(range(0, int(datadf2['Time (seconds)'].values.max()), 5))], mode='lines', line={'dash': 'dash'}, name=f'Threshold ({threshold}mV)'))

		# Draw rectangles around each continuous or discrete event
		for start, end in zip(merged_starts, merged_ends):
			# Adjust end for discrete events
			if end == start:
				end = start + window_size  # Adjust to mark as full window size for discrete events

			fig.add_shape(
				type="rect",
				x0=start, y0=-2, x1=end, y1=2,
				line=dict(color="LightSalmon"),
				fillcolor="Salmon",
				opacity=0.5
			)

		fig.update_layout(template="plotly_dark", title=fig_nammer, xaxis_title='Time (seconds)', yaxis_title='Signal (mV)')
		fig.write_html(main_dir + f'\\Traces\\{fig_nammer}.html')

		# Store events into the DataFrame
		mbe_builder = {'Begin': merged_starts, 'End':merged_ends, 'Duration': [end - start for start, end in zip(merged_starts, merged_ends)]}
		de_builder = {'Flagged Event': master_peaks}

		mbedf = pd.DataFrame(mbe_builder)
		dedf = pd.DataFrame(de_builder)
		channel_df = pd.concat([mbedf, dedf], ignore_index=True, axis=1)
		channel_df.columns = [f'Start - Ch.{pos+1}', f'End - Ch.{pos+1}', f'Duration - Ch.{pos+1}', f'Flagged Event - Ch.{pos+1}']
		master_df = pd.concat([master_df, channel_df], axis=1)


	
	with pd.ExcelWriter(main_dir + f'\\Output\\Data from {fig_name}.xlsx') as writer:
		master_df.to_excel(writer)


def spike_parameters():
	mainwb = xw.Book.caller()
	infosheet = 'SpikeSPY Parameters'
	master_df = pd.DataFrame()
	datafolder = main_dir + '\\Input'
	window_size = int(mainwb.sheets[infosheet].range('B2').value)
	slide_size = mainwb.sheets[infosheet].range('B3').value
	low_freq = mainwb.sheets[infosheet].range('B4').value
	high_freq = mainwb.sheets[infosheet].range('B5').value
	threshold = mainwb.sheets[infosheet].range('B6').value
	reset_threshold = mainwb.sheets[infosheet].range('B7').value
	graph_downsample = int(mainwb.sheets[infosheet].range('B8').value)
	
	
	file_name = mainwb.sheets[infosheet].range('B1').value
	if file_name == 'All':
		for file in os.listdir(datafolder):
			test = bioread.read_file(datafolder + f'\\{file}')
			SpikeSpy(file_name = file, test=test, datafolder = datafolder, low_freq = low_freq, high_freq = high_freq, threshold = threshold, graph_downsample = graph_downsample, master_df = master_df, window_size = window_size, slide_size = slide_size, reset_threshold= reset_threshold)
	else:
		test = bioread.read_file(datafolder + f'\\{file_name}')
		SpikeSpy(file_name = file_name, test=test, datafolder = datafolder, low_freq = low_freq, high_freq = high_freq, threshold = threshold, graph_downsample = graph_downsample, master_df = master_df, window_size = window_size, slide_size = slide_size , reset_threshold= reset_threshold)

	
