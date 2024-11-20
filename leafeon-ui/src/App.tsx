import { useEffect, useRef, useState } from 'react';
import { ReactSketchCanvas, ReactSketchCanvasRef } from 'react-sketch-canvas';
import { Button } from './components/ui/button';
import { ChartConfig, ChartContainer } from './components/ui/chart';
import { Bar, BarChart } from 'recharts';
import { get } from 'http';
import { Chart, ChartData } from './components/chart';
import {
	Card,
	CardContent,
	CardDescription,
	CardHeader,
	CardTitle,
} from './components/ui/card';

function App() {
	const [chartData, setChartData] = useState<ChartData | undefined>(
		undefined
	);
	const [label, setLabel] = useState('');

	return (
		<main className="h-screen w-screen m-4 flex gap-2">
			<Card className="h-min">
				<CardHeader>
					<CardTitle>Leafeon ui</CardTitle>
					<CardDescription>Draw a digit</CardDescription>
				</CardHeader>
				<CardContent>
					<Canvas setChartData={setChartData} setLabel={setLabel} />
				</CardContent>
			</Card>
			{chartData && <Chart chartData={chartData} label={label} />}
		</main>
	);
}

export default App;

function Canvas({
	setChartData,
	setLabel,
}: {
	setChartData: (data: any) => void;
	setLabel: (label: string) => void;
}) {
	const canvasRef = useRef<ReactSketchCanvasRef>(null);
	const [waiting, setWaiting] = useState(false);

	const getData = async () => {
		if (waiting) return;
		const value = await canvasRef.current?.exportImage('png');
		if (!value) return;
		setWaiting(true);
		const res = await fetch('http://localhost:3000/predict', {
			body: value,
			method: 'POST',
		}).then(res => res.json());
		setWaiting(false);
		const confidence = (res.confidence as number[]).map((x, i) => ({
			confidence: x * 100.0,
			label: i,
		}));
		setChartData(confidence);
		setLabel(res.label);
		console.log(res.label);
	};

	useEffect(() => {
		if (!waiting) {
			getData();
		}
	}, [waiting]);

	return (
		<div className="flex flex-col gap-2">
			<div className="flex">
				<Button onClick={() => canvasRef.current?.clearCanvas()}>
					reset
				</Button>
			</div>
			<div style={{ imageRendering: 'pixelated' }}>
				<ReactSketchCanvas
					width="280px"
					height="280px"
					strokeColor="black"
					canvasColor="white"
					strokeWidth={24}
					ref={canvasRef}
					onChange={() => {
						getData();
					}}
				/>
			</div>
		</div>
	);
}
