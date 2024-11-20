'use client';

import { TrendingUp } from 'lucide-react';
import { Bar, BarChart, XAxis, YAxis } from 'recharts';

import {
	Card,
	CardContent,
	CardDescription,
	CardFooter,
	CardHeader,
	CardTitle,
} from '@/components/ui/card';
import {
	ChartConfig,
	ChartContainer,
	ChartTooltip,
	ChartTooltipContent,
} from '@/components/ui/chart';

const chartConfig = {
	confidence: {
		theme: {
			dark: 'white',
			light: 'black',
		},
		label: 'Confidence',
	},
} satisfies ChartConfig;

export type ChartData = {
	confidence: number;
	label: number;
}[];

export function Chart({
	label,
	chartData,
}: {
	label: string;
	chartData: ChartData;
}) {
	return (
		<Card className="h-min">
			<CardHeader>
				<CardTitle>🧙‍♂️ That's clearly a {label}</CardTitle>
			</CardHeader>
			<CardContent>
				<ChartContainer config={chartConfig} className="min-h-64">
					<BarChart
						accessibilityLayer
						data={chartData}
						layout="vertical"
						margin={{
							left: 0,
						}}
					>
						<YAxis
							dataKey="label"
							type="category"
							tickLine={false}
							tickMargin={10}
							axisLine={false}
						/>
						<XAxis dataKey="confidence" type="number" hide />
						<ChartTooltip
							cursor={false}
							content={<ChartTooltipContent hideLabel />}
						/>
						<Bar
							dataKey="confidence"
							layout="vertical"
							fill="hsl(var(--chart-2))"
							radius={2}
						/>
					</BarChart>
				</ChartContainer>
			</CardContent>
			<CardFooter className="flex-col items-start gap-2 text-sm">
				<div className="flex gap-2 font-medium leading-none">
					{chartData
						.map(el => el.confidence)
						.reduce((a, b) => Math.max(a, b))
						.toPrecision(3)}
					% sure! <TrendingUp className="h-4 w-4" />
				</div>
				<div className="leading-none text-muted-foreground">
					Showing how confident the model is for every digit
				</div>
			</CardFooter>
		</Card>
	);
}
